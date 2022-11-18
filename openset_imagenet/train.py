import random
import time
import sys
import pathlib
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device
import vast
from loguru import logger
import tqdm
from .metrics import confidence
from .dataset import ImagenetDataset
from .model import ResNet50
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
from .perturbations import decay_epsilon, Noise, fgsm_attack


def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name(str): File name.
        model(torch module): Pytorch model.
        epoch(int): Current epoch.
        opt(torch optimizer): Current optimizer.
        best_score_(float): Current best score.
        scheduler(torch lr_scheduler): Pytorch scheduler.
    """
    # If model is DistributedDataParallel extracts the module.
    if isinstance(model, DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    data = {"epoch": epoch + 1,
            "model_state_dict": state,
            "opt_state_dict": opt.state_dict(),
            "best_score": best_score_}
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, f_name)


def load_checkpoint(model, checkpoint, opt=None, scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimizer): An optimizer to load the state dictionary. Defaults to None.
        scheduler (torch lr_scheduler): Learning rate scheduler. Defaults to None.
    """
    file_path = pathlib.Path(checkpoint)
    if file_path.is_file():  # First check if file exists
        # breakpoint()
        checkpoint = torch.load(file_path, map_location=vast.tools._device)
        key = list(checkpoint["model_state_dict"].keys())[0]
        # If module was saved as DistributedDataParallel then removes the world "module"
        # from dictionary keys
        if key[:6] == "module":
            new_state_dict = OrderedDict()
            for k_i, v_i in checkpoint["model_state_dict"].items():
                key = k_i[7:]  # remove "module"
                new_state_dict[key] = v_i
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if opt is not None:  # Load optimizer state
            opt.load_state_dict(checkpoint["opt_state_dict"])

        if scheduler is not None:  # Load scheduler state
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        return start_epoch, best_score
    raise Exception(f"Checkpoint file '{checkpoint}' not found")


def train(model, data_loader, optimizer, loss_fn, trackers, cfg, noise_gen=None):
    """Main training loop."""

    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    j, j_kn, j_neg = None, None, None

    # Calculate label of perturbed negatives
    if cfg.loss.type == 'garbage' and cfg.adv.clean_neg:
        negative_label = data_loader.dataset.unique_classes[-1]
    elif cfg.loss.type == 'garbage' and not cfg.adv.clean_neg:
        negative_label = data_loader.dataset.unique_classes[-1]+1
    else:
        negative_label = -1

    for images, labels in data_loader:
        model.train()  # To collect batch-norm statistics
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad(set_to_none=True)
        images = device(images)
        labels = device(labels)

        # If the gradient with respect to the input is needed
        if cfg.adv.who == 'fgsm':
            images.requires_grad_()
            images.grad = None

        # Forward pass
        logits, _ = model(images)

        # Separate positives from negatives to keep track of separated losses
        if cfg.loss.type == 'entropic':
            kn_idx = labels >= 0
            neg_idx = ~kn_idx
            if torch.all(kn_idx):       # only knowns
                j = loss_fn(logits, labels)
                trackers['j_kn'].update(j.item(), batch_len)
            elif torch.all(neg_idx):    # only negatives
                j = loss_fn(logits, labels)
                trackers['j_neg'].update(j.item(), batch_len)
            else:                       # both known and negatives
                j_kn = loss_fn(logits[kn_idx], labels[kn_idx])
                j_neg = loss_fn(logits[neg_idx], labels[neg_idx])
                j = j_kn + j_neg
                trackers['j_kn'].update(j_kn.item(), batch_len)
                trackers['j_neg'].update(j_neg.item(), batch_len)
        else:
            j = loss_fn(logits, labels)
            trackers['j'].update(j.item(), batch_len)

        # Backward pass
        j.backward()

        if cfg.adv.who == 'no_adv':
            optimizer.step()
        else:
            # Steps: Select samples to perturb, create perturbed samples
            # calculate loss of perturbed samples, backward pass
            model.eval()  # Stop batch normalisation statistics

            # Get the candidates to adversarial samples
            num_corr_samples = None
            correct_idx = None

            if cfg.adv.mode == 'filter':  # Perturb corrected classified samples
                correct_idx = filter_correct(logits, labels, cfg.threshold, negative_label)
                num_corr_samples = len(correct_idx[0])
            elif cfg.adv.mode == 'full':  # Perturb all samples
                correct_idx = torch.arange(batch_len, device=vast.tools._device)
                num_corr_samples = batch_len
            trackers['n_pert'].update(num_corr_samples)
            # Create perturbed samples
            if num_corr_samples > 0:
                im_corr = images[correct_idx]

                if cfg.adv.who == 'fgsm':
                    im_corr_grad = images.grad[correct_idx]
                    im_pert, label_pert = fgsm_attack(clean_im=im_corr,
                                                      epsilon=cfg.adv.epsilon,
                                                      grad=im_corr_grad,
                                                      negs_label=negative_label,
                                                      device=vast.tools._device)
                else:
                    im_pert = noise_gen.perturb(clean_im=im_corr)
                    label_pert = noise_gen.get_labels(shape=im_corr.shape[0],
                                                      device=vast.tools._device,
                                                      negs_label=negative_label)

                # forward pass with perturbed samples
                logits, _ = model(im_pert)
                j_pert = loss_fn(logits, label_pert)
                trackers['j_pert'].update(j_pert.item(), num_corr_samples)
                j_pert.backward()
            optimizer.step()


def validate(model, data_loader, loss_fn, n_classes, trackers, cfg):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
        loss_fn: Loss function
        n_classes(int): Total number of classes
        trackers(dict): Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset all validation metrics
    for metric in trackers.values():
        metric.reset()

    if cfg.loss.type == "garbage":
        min_unk_score = 0.
        unknown_class = n_classes - 1
        last_valid_class = -1
    else:
        min_unk_score = 1. / n_classes
        unknown_class = -1
        last_valid_class = None

    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, n_classes), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, _ = model(images)
            scores = torch.nn.functional.softmax(logits, dim=1)

            # Separate positive from negative to keep track of the loss
            if cfg.loss.type == 'entropic':
                kn_idx = labels >= 0
                neg_idx = ~kn_idx
                if torch.all(kn_idx):  # only knowns
                    j = loss_fn(logits, labels)
                    trackers['j_kn'].update(j.item(), batch_len)
                elif torch.all(neg_idx):  # only negatives
                    j = loss_fn(logits, labels)
                    trackers['j_neg'].update(j.item(), batch_len)
                else:  # both known and negatives
                    j_kn = loss_fn(logits[kn_idx], labels[kn_idx])
                    j_neg = loss_fn(logits[neg_idx], labels[neg_idx])
                    j = j_kn + j_neg
                    trackers['j_kn'].update(j_kn.item(), batch_len)
                    trackers['j_neg'].update(j_neg.item(), batch_len)
            else:
                j = loss_fn(logits, labels)
                trackers['j'].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            start_ix = i * cfg.batch_size
            all_targets[start_ix: start_ix + batch_len] = labels
            all_scores[start_ix: start_ix + batch_len] = scores

        kn_conf, kn_count, neg_conf, neg_count = confidence(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score,
            unknown_class=unknown_class,
            last_valid_class=last_valid_class)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


def predict(scores, threshold, neg_label=-1):
    """ Returns predicted score and label based on the softmax scores. When score < threshold, the
    sample is labeled as unk_label.

    Args:
        scores: Softmax scores of all classes and samples in batch.
        threshold: Minimum score to classify as a known sample.
        neg_label: Label reserved for negatives samples.

    Returns:
        Tuple with predicted class and score.
    """
    with torch.no_grad():
        pred_score, pred_class = torch.max(scores, dim=1)
        unk = pred_score < threshold
        pred_class[unk] = neg_label
        return pred_score, pred_class


def filter_correct(logits, targets, threshold, neg_label=-1):
    """Returns the indices of correctly predicted known samples.

    Args:
        logits (tensor): Logits tensor
        targets (tensor): Targets tensor
        threshold (float): Minimum score for the target to be classified as known.
        neg_label (int): Label reserved for negative samples.

    Returns:
        tuple: Contains in fist position a tensor with indices of correctly predicted samples.
    """
    with torch.no_grad():
        scores = torch.nn.functional.softmax(logits, dim=1)
        _, pred_class = predict(scores, threshold,  neg_label)
        correct = (targets != neg_label) * (pred_class == targets)
        return torch.nonzero(correct, as_tuple=True)


def get_arrays(model, loader):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
    """
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model.logits.out_features  # logits output classes
        features_dim = model.logits.in_features  # features dimensionality
        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, logits_dim), device="cpu")

        index = 0
        for images, labels in tqdm.tqdm(loader):
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logit, feature = model(images)
            score = torch.nn.functional.softmax(logit, dim=1)
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logit.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = score.detach().cpu()
            index += curr_b_size
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())


def worker(cfg):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    # set_seeds(cfg.seed)

    BEST_SCORE = 0.0    # Best validation score
    START_EPOCH = 0     # Initial training epoch

    # Configure logger. Log only on first process. Validate only on first process.
    # msg_format = "{time:DD_MM_HH:mm} {message}"
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink=cfg.output_directory / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    # Set image transformations
    train_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(0.5),
         transforms.ToTensor()])

    val_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    # create datasets
    train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
    val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr)

        val_ds = ImagenetDataset(
            csv_file=val_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=val_tr)
        initial_classes = train_ds.label_count

        # Setting labels for every training case
        if cfg.loss.type == 'softmax' or not cfg.adv.clean_neg:
            train_ds.remove_negative_label()
        elif cfg.loss.type == 'garbage':
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
    else:
        raise FileNotFoundError("train/validation file does not exist")

    # determine number of classes
    if cfg.loss.type == "entropic":
        # number of classes - 1 since entropic doesn't add new class for negative or unknown samples
        n_classes = initial_classes - 1
    else:
        # num of classes when training with extra garbage class or when unknowns are removed
        n_classes = initial_classes

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,)

    # setup device
    if cfg.gpu is not None:
        set_device_gpu(index=cfg.gpu)
    else:
        logger.warning("No GPU device selected, training will be extremely slow")
        set_device_cpu()

    # Callbacks
    early_stopping = None
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # Setting loss
    if cfg.loss.type == "entropic":
        # We select entropic loss using the unknown class weights from the config file
        loss = EntropicOpensetLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == "softmax":
        # We need to ignore the index only for validation loss computation
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif cfg.loss.type == "garbage":
        # We use balanced class weights
        class_weights = device(train_ds.calculate_class_weights())
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
    device(model)

    # Create optimizer
    if cfg.opt.type == "sgd":
        opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)

    # Learning rate scheduler
    if cfg.opt.decay > 0:
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg.opt.decay,
            gamma=cfg.opt.gamma,
            verbose=True)
    else:
        scheduler = None

    # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        if cfg.train_mode == "finetune": # TODO: Simplify the modes, finetune is not necessary
            START_EPOCH, _ = load_checkpoint(
                model=model,
                checkpoint=cfg.checkpoint,
                opt=None,
                scheduler=None)
            BEST_SCORE = 0
        else:
            START_EPOCH, BEST_SCORE = load_checkpoint(
                model=model,
                checkpoint=cfg.checkpoint,
                opt=opt,
                scheduler=scheduler)
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")

    # Print info to console and setup summary writer
    logger.info("============ Data ============")
    logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
    logger.info(f"General mode: {cfg.train_mode}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"workers: {cfg.workers}")
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"optimizer: {cfg.opt.type}")
    logger.info(f"Learning rate: {cfg.opt.lr}")
    logger.info(f"Device: {vast.tools._device}")
    logger.info("======== Perturbations ========")
    logger.info(f"Perturbations: {cfg.adv.who}")
    if cfg.adv.who in "fgsm, bernoulli":
        logger.info(f"Epsilon: {cfg.adv.epsilon}")
    if cfg.adv.who == "gaussian":
        logger.info(f"Noise~N(0, {cfg.adv.std})")
    if cfg.adv.who == "uniform":
        logger.info(f"Noise~U[{cfg.adv.low}, {cfg.adv.high})")
    if cfg.adv.who == "bernoulli":
        logger.info(f"Noise~Bernoulli({cfg.adv.p})")
    logger.info(f"Perturb. mode: {cfg.adv.mode}")
    logger.info(f"Decay factor mu: {cfg.adv.mu}")
    logger.info(f"Decay every {cfg.adv.decay} epochs")
    logger.info(f"Include clean negatives: {cfg.adv.clean_neg}")
    logger.info("======== Training ========")
    writer = SummaryWriter(log_dir=cfg.output_directory, filename_suffix="-"+cfg.log_name)

    # Setup training with perturbations
    start_epsilon = cfg.adv.epsilon
    initial_who = cfg.adv.who
    noise_gen = None
    if initial_who == 'gaussian':
        noise_gen = Noise(initial_who, loc=0, std=cfg.adv.std)
    elif initial_who == 'bernoulli':
        noise_gen = Noise(initial_who, prob=cfg.adv.p)
    elif initial_who == 'uniform':
        noise_gen = Noise(initial_who, low=cfg.adv.low, high=cfg.adv.high)

    if cfg.adv.wait > 0:  # train without adv for a nr. of epochs then add adversarial samples.
        cfg.adv.who = 'no_adv'

    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()

        # when to add perturbed samples
        if (cfg.adv.wait > 0) and (epoch >= cfg.adv.wait):  # After wait epochs add pert. samples
            cfg.adv.who = initial_who

        # If using epsilon with decay
        if (cfg.adv.who in ['fgsm', 'bernoulli']) and 0 < cfg.adv.mu < 1 and cfg.adv.decay > 0:
            cfg.adv.epsilon = decay_epsilon(start_epsilon, cfg.adv.mu, epoch, cfg.adv.decay)
            logger.info(f"current epsilon:{cfg.adv.epsilon}")

        # training loop
        train(
            model=model,
            data_loader=train_loader,
            optimizer=opt,
            loss_fn=loss,
            trackers=t_metrics,
            cfg=cfg,
            noise_gen=noise_gen)

        train_time = time.time() - epoch_time

        # validation loop
        validate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss,
            n_classes=n_classes,
            trackers=v_metrics,
            cfg=cfg)

        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        if cfg.loss.type == 'entropic':
            writer.add_scalar("train/loss_kn", t_metrics["j_kn"].avg, epoch)
            writer.add_scalar("train/loss_neg", t_metrics["j_neg"].avg, epoch)
            writer.add_scalar("val/loss_kn", v_metrics["j_kn"].avg, epoch)
            writer.add_scalar("val/loss_neg", v_metrics["j_neg"].avg, epoch)
        else:
            writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
            writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        if cfg.adv.who != 'no_adv':
            writer.add_scalar('train/perturbed', t_metrics['j_pert'].avg, epoch)
            writer.add_scalar('val/perturbed', v_metrics['j_pert'].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        #  training information on console
        # validation+metrics writer+save model time
        val_time = time.time() - train_time - epoch_time

        def pretty_print(d):
            return dict(d)

        logger.info(
            f"loss:{cfg.loss.type} "
            f"protocol:{cfg.protocol} "
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")

        # save best model and current model
        ckpt_name = str(cfg.output_directory / cfg.name) + "_curr.pth"
        save_checkpoint(ckpt_name, model, epoch, opt, curr_score, scheduler=scheduler)

        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = str(cfg.output_directory / cfg.name) + "_best.pth"
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model {ckpt_name} at epoch: {epoch}")
            save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    # clean everything
    del model
    logger.info("Training finished")
