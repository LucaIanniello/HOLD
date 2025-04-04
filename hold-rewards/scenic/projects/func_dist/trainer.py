"""Training script for Functional Distances."""

import copy
import functools
import gzip
import os
import pickle
from typing import Any, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.projects.func_dist import model_utils as func_dist_model_utils
from scenic.projects.func_dist import train_utils as func_dist_train_utils
from scenic.projects.func_dist import model as func_dist_model
from scenic.projects.func_dist import holdc_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    test_dataset: Optional[dataset_utils.Dataset] = None,
    workdir: str,
    writer: metric_writers.MetricWriter,
    debug: bool = False,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  is_multilabel_model = (config.model_name == 'vivit_multilabel_classification')
  get_confusion_matrix = (config.get('confusion_matrix_metrics', False)
                          and not is_multilabel_model)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  # if (start_step == 0  # Which means "no" checkpoint is restored!
  #     and config.get('init_from') is not None):
  #   restored_model_cfg = config.init_from.get('model_config')
  #   init_checkpoint_path = config.init_from.get('checkpoint_path')
  #   checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
  #   if checkpoint_format == 'scenic':
  #     restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
  #         init_checkpoint_path, train_state, assert_exist=True)
  #   elif checkpoint_format == 'big_vision':
  #     restored_train_state = pretrain_utils.convert_big_vision_to_scenic_checkpoint(
  #         init_checkpoint_path, train_state)
  #     # Config dict in big_vision is not the same format as scenic.
  #     # Therefore, make sure config match the config of the loaded model!
  #     restored_model_cfg = copy.deepcopy(config)
  #     # The following is needed when the restored and target models used a
  #     # different classifier. As big_vision uses a different config dict, we
  #     # have to specify this manually.
  #     restored_model_cfg.model.classifier = config.init_from.get(
  #         'classifier_type', 'token')

  #   train_state = model.init_from_train_state(train_state, restored_train_state,
  #                                             restored_model_cfg)
  #   # Free unnecessary memory.
  #   del restored_train_state
  # elif start_step == 0:
    
  logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  train_step_pmapped = jax.pmap(
      functools.partial(
          vivit_train_utils.train_step,
          flax_model=model.flax_model,
          learning_rate_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          return_logits=debug,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          vivit_train_utils.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          return_logits_and_labels=debug,
          return_confusion_matrix=get_confusion_matrix,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  print('Steps per epoch', steps_per_epoch, 'log_eval_steps', log_eval_steps)
  log_test_steps = 0
  if config.dataset_configs.get('eval_test_metrics'):
    log_test_steps = int(steps_per_epoch *
                         config.dataset_configs.log_test_epochs)
    test_batch_size = config.dataset_configs.get('test_batch_size')

    test_step_pmapped = jax.pmap(
        functools.partial(
            func_dist_train_utils.full_seq_test_step,
            flax_model=model.flax_model,
            num_stacked_frames=config.dataset_configs.get('num_frames') - 1,
            metrics_fn=model.get_metrics_fn('test'),
            test_batch_size=test_batch_size,
            return_logits_and_labels=is_multilabel_model,
            return_confusion_matrix=get_confusion_matrix,
            # n_clips=config.get('multicrop_clips_per_device', 2),
            debug=config.debug_eval),
        axis_name='batch',
        # We can donate the test_batch's buffer.
        # donate_argnums=(1,),
    )

    # assert config.dataset_configs.test_batch_size == jax.local_device_count(), (
    #     'The per-host batch size must be equal to the number of local devices.'
    #     'This ensures that each TPU device is processing different views of'
    #     'the same original video.')

    videos_per_test_step = 1
    total_test_steps = int(
        np.ceil(dataset.meta_data['num_test_examples'] /
                (videos_per_test_step *
                 config.get('dataset_configs.num_test_clips') *
                 jax.process_count())))
    steps_per_test = config.get('steps_per_test') or total_test_steps

  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = [report_progress]
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  # Manually defragment memory before starting training, if we are using the
  # tfrt runtime.
  do_memory_defrag = False
  if config.get('do_memory_defrag', False):
    client = jax.lib.xla_bridge.get_backend()
    try:
      logging.info('Defragmenting memory')
      client.defragment()
      do_memory_defrag = True
    except RuntimeError:
      logging.warn('Memory defragmentation not possible, use the tfrt runtime')

  best_val_metrics = func_dist_train_utils.load_json_metrics(workdir, 'val')
  best_test_metrics = func_dist_train_utils.load_json_metrics(workdir, 'test')
  # Evaluation metrics where lower is better. Higher is better is assumed for 
  # the rest.
  loss_metrics = [
      'mean_absolute_error', 'mean_squared_error',
      'mean_absolute_error_steps', 'mean_squared_error_steps',
      'mean_absolute_error_seconds', 'mean_squared_error_seconds',
      'triplet_svtc_loss',
      'misclassification_rate', 'hinge_loss',
  ]
  n_ckpts_to_keep = config.get('num_training_epochs', 5)
  logging.info(f'Num ckpts to keep: {n_ckpts_to_keep}')

  train_vid_id_to_indices = {}
  valid_vid_id_to_indices = {}
  frame_rates_per_video = {}
  len_in_steps_per_video = {}
  len_in_secs_per_video = {}
  valid_len_in_secs_per_video = {}
  train_indices_match = 0
  train_indices_mismatch = 0
  train_vid_ids = set()
  valid_vid_ids = set()
  test_vid_ids = set()
  if debug:
    traindir = os.path.join(workdir, 'train')
    valdir = os.path.join(workdir, 'eval')
    testdir = os.path.join(workdir, 'test')
    for outdir in [traindir, valdir, testdir]:
      if not os.path.exists(outdir):
        os.makedirs(outdir)

  additional_metrics_fn = functools.partial(
      func_dist_model.temporal_regression_metrics_function,
      metrics=immutabledict({
          'spearman_correlation': (
              func_dist_model_utils.spearman_correlation,
              holdc_model.num_videos)
      }),
      split='test',
      pmapped=False)

  for step in range(start_step + 1, total_steps + 1):
    epoch = step // steps_per_epoch + 1
    step_in_epoch = step % steps_per_epoch + 1
    if step % steps_per_epoch == 0:
        print('Starting epoch', step // steps_per_epoch + 1)
        print('Train video ids', len(train_vid_ids))
        print('Valid video ids', len(valid_vid_ids))
        print('Test video ids', len(test_vid_ids))
        mean_rate = np.mean(list(frame_rates_per_video.values()))
        mean_len_steps = np.mean(list(len_in_steps_per_video.values()))
        mean_len_secs = np.mean(list(len_in_secs_per_video.values()))
        print('Average frame rate', mean_rate)
        print('Average length', mean_len_steps, f'({mean_len_secs} seconds)')
        total_len_secs = np.sum(list(len_in_secs_per_video.values()))
        total_val_len_secs = np.sum(list(valid_len_in_secs_per_video.values()))
        print(f'Total length: {total_len_secs} seconds')
        print(f'Total val length: {total_val_len_secs} seconds')
        step_lengths = np.array(list(len_in_steps_per_video.values()))
        mean_target_steps = np.mean((step_lengths + 1) / 3)
        sec_lengths = np.array(list(len_in_secs_per_video.values()))
        mean_target_secs = np.mean((sec_lengths + 1) / 3)
        print('Average distance target', mean_target_steps,
              f'({mean_target_secs} seconds)')
        print('Total number of videos', len(frame_rates_per_video))
    with jax.profiler.StepTraceContext('train', step_num=step):
      train_batch = next(dataset.train_iter)
      for video_id in train_batch['video_id'][0]:
          train_vid_ids.add(int(video_id))

      for vid_id, vid_len, frame_rate in zip(
          train_batch['video_id'][0], train_batch['video_length'][0],
          train_batch['frame_rate'][0]):
          frame_rates_per_video[vid_id.item()] = frame_rate[0].item()
          len_in_steps_per_video[vid_id.item()] = vid_len.item()
          len_in_secs_per_video[vid_id.item()] = (
              vid_len.item() / frame_rate[0].item())
      for video_id, targets, indices, video_length in zip(
          train_batch['video_id'][0], train_batch['targets'][0],
          train_batch['indices'][0], train_batch['video_length'][0]):
        new_indices = indices.tolist() + [targets[0], video_length]
        video_id = int(video_id)
        if video_id in train_vid_id_to_indices:
          prev_indices = train_vid_id_to_indices[video_id]
          if prev_indices == new_indices:
            # print('Train indices', video_id, 'match', new_indices, 'vs',
            #       prev_indices)
            train_indices_match += 1
          else:
            train_indices_mismatch += 1
        else:
          train_vid_id_to_indices[video_id] = new_indices

      for vid_idx, video_id in enumerate(train_batch['video_id'][0]):
        video_id = int(video_id)
        if video_id == 84374:
          print('\n\nTrain step', step, '- sampled video', video_id, 'with',
                train_batch['indices'][0, vid_idx], train_batch['targets'][0, vid_idx],
                train_batch['video_length'][0, vid_idx], '\n\n')

      # TODO: Move these transformations to the input pipeline.
      if (train_batch['inputs'].shape[2]
          != config.dataset_configs.get('num_frames')):
        for k in ['inputs', 'targets', 'targets_seconds']:
          input_size = int(train_batch[k].shape[2] / 2)
          train_batch[k] = jnp.concatenate(
              [train_batch[k][:, :, :input_size],
               train_batch[k][:, :, input_size:]], axis=1)
        for k in ['frame_rate', 'batch_mask']:
          train_batch[k] = jnp.concatenate(
              [train_batch[k], train_batch[k]], axis=1)
      train_returns = train_step_pmapped(train_state, train_batch)
      if debug:
        train_state, t_metrics, lr, logits = train_returns
      else:
        train_state, t_metrics, lr = train_returns
      if debug and epoch < 4:
        with gzip.open(os.path.join(traindir, f'train_batch_e{epoch}_s{step_in_epoch}.pkl'), 'wb') as f:
          pickle.dump(train_batch, f)
        with gzip.open(os.path.join(traindir, f'train_preds_e{epoch}_s{step_in_epoch}.pkl'), 'wb') as f:
          pickle.dump(logits, f)
        with gzip.open(os.path.join(traindir, f'train_metrics_e{epoch}_s{step_in_epoch}.pkl'), 'wb') as f:
          pickle.dump(train_utils.unreplicate_and_get(t_metrics), f)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': lr})

      # if step % 50 == 0:
      #   if train_indices_match > 0 or train_indices_mismatch > 0:
      #     print('Train indices match', train_indices_match, 'vs mismatch',
      #           train_indices_mismatch)

    for h in hooks:
      # Catch exception in case XProf fails.
      try:
        h(step)
      except ValueError as error:
        logging.exception('Hook failed: %r', error)

    chrono.pause()  # Below are once-in-a-while ops -> pause.
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer=writer)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_map(train_utils.unreplicate_and_get,
                                     train_metrics),
          extra_training_logs=jax.tree_map(train_utils.unreplicate_and_get,
                                           extra_training_logs),
          writer=writer,
          key_separator='/')
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      if do_memory_defrag:
        logging.info('Defragmenting memory')
        client.defragment()

    ################### EVALUATION ################################
    if (step % log_eval_steps == 1) or (step == total_steps):
      logging.info('Running evaluation')
      print(step, log_eval_steps, step % log_eval_steps)
      with report_progress.timed('eval'):
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

        eval_metrics = []
        additional_summary = None
        if is_multilabel_model:
          eval_logits = []
          eval_labels = []
          n_classes = dataset.meta_data['num_classes']
        if get_confusion_matrix:
          confusion_matrices = []
          n_classes = dataset.meta_data['num_classes']

        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for eval_step in range(steps_per_eval):
          if eval_step % (max(steps_per_eval // 10, 5)) == 0:
            logging.info(f'Starting eval step {eval_step} / {steps_per_eval}')
          eval_batch = next(dataset.valid_iter)
          if (eval_batch['inputs'].shape[2]
              != config.dataset_configs.get('num_frames')):
            for k in ['inputs', 'targets', 'targets_seconds']:
              input_size = int(eval_batch[k].shape[2] / 2)
              eval_batch[k] = jnp.concatenate(
                  [eval_batch[k][:, :, :input_size],
                   eval_batch[k][:, :, input_size:]], axis=1)
            for k in ['frame_rate', 'batch_mask']:
              eval_batch[k] = jnp.concatenate(
                  [eval_batch[k], eval_batch[k]], axis=1)
          for video_id in eval_batch['video_id'][0]:
              valid_vid_ids.add(int(video_id))
          for vid_id, vid_len, frame_rate in zip(
              eval_batch['video_id'][0], eval_batch['video_length'][0],
              eval_batch['frame_rate'][0]):
              valid_len_in_secs_per_video[vid_id.item()] = (
                  vid_len.item() / frame_rate[0].item())
          for video_id, targets, indices, video_length in zip(
              eval_batch['video_id'][0], eval_batch['targets'][0], eval_batch['indices'][0], eval_batch['video_length'][0]):
            new_indices = indices.tolist() + [targets[0], video_length]
            if video_id in valid_vid_id_to_indices:
              prev_indices = valid_vid_id_to_indices[video_id]
              if not prev_indices == new_indices:
                print('Validation indices do not match', new_indices, 'vs',
                      prev_indices)
            else:
              valid_vid_id_to_indices[video_id] = new_indices
          e_metrics = eval_step_pmapped(train_state, eval_batch)
          if debug:
            e_metrics, logits_batch, labels_batch = e_metrics
          if debug and epoch < 4:
            with gzip.open(os.path.join(valdir, f'val_batch_e{epoch}_s{eval_step}.pkl'), 'wb') as f:
              pickle.dump(eval_batch, f)
            with gzip.open(os.path.join(valdir, f'val_preds_e{epoch}_s{eval_step}.pkl'), 'wb') as f:
              pickle.dump(logits_batch, f)
            with gzip.open(os.path.join(valdir, f'val_metrics_e{epoch}_s{eval_step}.pkl'), 'wb') as f:
              pickle.dump(e_metrics, f)

          if is_multilabel_model:
            e_metrics, logits_batch, labels_batch = e_metrics
            eval_logits.append(vivit_train_utils.to_cpu(logits_batch))
            eval_labels.append(vivit_train_utils.to_cpu(labels_batch))
          if get_confusion_matrix:
            e_metrics, conf_matrix = e_metrics
            confusion_matrices.append(vivit_train_utils.to_cpu(conf_matrix))
          # Fetch e_metrics to host and store.
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

        # Compute global metrics if applicable from all the batches.
        if is_multilabel_model:
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(eval_logits, axis=0),
              np.concatenate(eval_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        if get_confusion_matrix:
          additional_summary = evaluation_lib.compute_confusion_matrix_metrics(
              confusion_matrices, return_per_class_metrics=n_classes < 10)
          if lead_host:
            conf_matrix_image = vivit_train_utils.render_confusion_matrices(
                confusion_matrices, normalization_method='rows')
            conf_matrix_unnorm = vivit_train_utils.render_confusion_matrices(
                confusion_matrices, normalization_method='none')

            writer.write_images(
                step, {'valid/conf_matrix': conf_matrix_image,
                       'valid/conf_matrix_unnormalized': conf_matrix_unnorm})

        # Log eval summary.
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=additional_summary,
            writer=writer,
            key_separator='/')
        writer.flush()

        if config.checkpoint:
          func_dist_train_utils.keep_best_checkpoints(eval_summary,
              best_val_metrics, n_ckpts_to_keep, loss_metrics, 'val', workdir,
              train_state, step, lead_host, chrono, report_progress)
        del eval_metrics
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

    ##################### CHECKPOINTING ###########################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    ############# TESTING ############################
    if (config.dataset_configs.get('eval_test_metrics') and
        ((step % log_test_steps == 1 and step > 1) or step == total_steps)):
      with report_progress.timed('test'):
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

        test_metrics = []
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)

        # At the end of training, evaluate on the whole test set.
        if step == total_steps:
          steps_per_test = total_test_steps

        logging.info('Starting test')
        for test_step in range(steps_per_test):
          test_batch = next(dataset.test_iter)
          for video_id in test_batch['video_id'][0]:
            test_vid_ids.add(int(video_id))
          if step == log_test_steps + 1 and test_step < 4:
            with open(
                os.path.join(workdir, f'test_batch_{step}.pkl'), 'wb') as f:
              pickle.dump(test_batch, f)
          test_batch['batch_mask'] = (
              func_dist_train_utils.mask_from_video_lengths(
                  jnp.transpose(test_batch['inputs'], (0, 2, 1, 3, 4, 5)),
                  test_batch['video_length']))
          test_batch['targets'] = jnp.transpose(
              test_batch['targets'], (0, 2, 1))
          if debug and epoch < 4:
            with gzip.open(os.path.join(testdir, f'test_batch_raw_e{epoch}_s{test_step}.pkl'), 'wb') as f:
              pickle.dump(test_batch, f)
          t_metrics, preds = test_step_pmapped(train_state, test_batch)
          additional_metrics = additional_metrics_fn(preds, test_batch)
          # Fetch t_metrics to host and store.
          t_metrics = train_utils.unreplicate_and_get(t_metrics)
          t_metrics = {**t_metrics, **additional_metrics}
          test_metrics.append(t_metrics)
          if debug and epoch < 4:
            with gzip.open(os.path.join(testdir, f'test_batch_e{epoch}_s{test_step}.pkl'), 'wb') as f:
              pickle.dump(test_batch, f)
            with gzip.open(os.path.join(testdir, f'test_preds_e{epoch}_s{test_step}.pkl'), 'wb') as f:
              pickle.dump(preds, f)
            with gzip.open(os.path.join(testdir, f'test_metrics_e{epoch}_s{test_step}.pkl'), 'wb') as f:
              pickle.dump(t_metrics, f)
        # Log eval summary.
        test_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=test_metrics,
            writer=writer,
            prefix='test',
            key_separator='/')
        logging.info('Completed test')
        writer.flush()

        if config.checkpoint:
          func_dist_train_utils.keep_best_checkpoints(test_summary,
              best_test_metrics, n_ckpts_to_keep, loss_metrics, 'test', workdir,
              train_state, step, lead_host, chrono, report_progress)

        # Free up some space.
        del test_metrics
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

    chrono.resume()  # un-pause now

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
