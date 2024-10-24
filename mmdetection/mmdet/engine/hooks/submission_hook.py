# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
import pandas as pd
import re

@HOOKS.register_module()
class SubmissionHook(Hook):
    """
    Hook for submitting results. Saves verification and test process prediction results.

    In the testing phase:

    1. Receives labels, scores, and bboxes from outputs and stores them in prediction_strings.
    2. Get the img_path of outputs and save it in file_names.

    Args:
        prediction_strings (list): [labels + ' ' + scores + ' ' + x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max]를 추가한 list
        file_names (list): img_path를 추가한 list
        test_out_dir (str) : 저장할 경로
    """

    def __init__(self, test_out_dir='submit'):
        self.test_outputs_data = []
        self.test_out_dir = test_out_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        
        for output in outputs:
            prediction_string = ''
            for label, score, bbox in zip(output.pred_instances.labels, output.pred_instances.scores, output.pred_instances.bboxes):
                bbox = bbox.cpu().numpy()
                # 이미 xyxy로 되어있음
                prediction_string += str(int(label.cpu())) + ' ' + str(float(score.cpu())) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' '
            match = re.search(r'test/(\d+\.jpg)', output.img_path)
            if match:
                self.test_outputs_data.append([int(match.group(1)[:4]), prediction_string, match.group(0)])
            else:
                assert 'File dir have Problem -- in Submission Hook'
        

    def after_test(self, runner: Runner):
        """
        Run after testing

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
        """
        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)
        self.test_outputs_data.sort(key=lambda x: x[0])
        
        prediction_strings = []
        file_names = []
        for _, predict, file_name in self.test_outputs_data:
            prediction_strings.append(predict)
            file_names.append(file_name)

        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        print(submission.head())
        submission.to_csv(osp.join(self.test_out_dir, 'submission.csv'), index=None)
        print('submission saved to {}'.format(osp.join(self.test_out_dir, 'submission.csv')))




# # Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
# from typing import Sequence

# from mmengine.hooks import Hook
# from mmengine.runner import Runner
# from mmengine.utils import mkdir_or_exist

# from mmdet.registry import HOOKS
# from mmdet.structures import DetDataSample
# import pandas as pd

# @HOOKS.register_module()
# class SubmissionHook(Hook):
#     """
#     Hook for submitting results. Saves verification and test process prediction results.

#     In the testing/validation phase:

#     1. Receives labels, scores, and bboxes from outputs and stores them in prediction_strings.
#     2. Get the img_path of outputs and save it in file_names.

#     Args:
#         output_dir (str): Directory name to save the submission files. Defaults to 'submit'.
#         mode (str): Mode of operation, either 'test' or 'val'. Defaults to 'test'.
#     """

#     def __init__(self, output_dir='submit', mode='test'):
#         self.outputs_data = []
#         self.output_dir = output_dir
#         if mode not in ['test', 'val']:
#             raise ValueError("mode must be either 'test' or 'val'")
#         self.mode = mode

#     def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
#                        outputs: Sequence[DetDataSample]) -> None:
#         """Run after every testing iteration.

#         Args:
#             runner (:obj:`Runner`): The runner of the testing process.
#             batch_idx (int): The index of the current batch in the test loop.
#             data_batch (dict): Data from dataloader.
#             outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
#                 that contain annotations and predictions.
#         """
#         self._process_outputs(outputs)

#     def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
#                       outputs: Sequence[DetDataSample]) -> None:
#         """Run after every validation iteration.

#         Args:
#             runner (:obj:`Runner`): The runner of the validation process.
#             batch_idx (int): The index of the current batch in the val loop.
#             data_batch (dict): Data from dataloader.
#             outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
#                 that contain annotations and predictions.
#         """
#         self._process_outputs(outputs)

#     def _process_outputs(self, outputs: Sequence[DetDataSample]) -> None:
#         """Process the outputs from each iteration and store them.

#         Args:
#             outputs (Sequence[:obj:`DetDataSample`]): Outputs from the model.
#         """
#         for output in outputs:
#             prediction_string = ''
#             labels = output.pred_instances.labels.cpu().numpy()
#             scores = output.pred_instances.scores.cpu().numpy()
#             bboxes = output.pred_instances.bboxes.cpu().numpy()

#             for label, score, bbox in zip(labels, scores, bboxes):
#                 prediction_string += f'{int(label)} {float(score)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} '

#             # Adjust the key to match how image paths are stored in your dataset
#             self.outputs_data.append([prediction_string.strip(), osp.basename(output.img_path)])

#     def after_test(self, runner: Runner):
#         """Run after the testing phase.

#         Args:
#             runner (:obj:`Runner`): The runner of the testing process.
#         """
#         self._save_submission(runner)

#     def after_val(self, runner: Runner):
#         """Run after the validation phase.

#         Args:
#             runner (:obj:`Runner`): The runner of the validation process.
#         """
#         self._save_submission(runner)

#     def _save_submission(self, runner: Runner):
#         """Save the submission CSV file.

#         Args:
#             runner (:obj:`Runner`): The runner of the process.
#         """
#         if self.output_dir is not None:
#             self.output_dir = osp.join(runner.work_dir, runner.timestamp, self.output_dir, self.mode)
#             mkdir_or_exist(self.output_dir)

#         prediction_strings = []
#         file_names = []
#         for predict, file_name in self.outputs_data:
#             prediction_strings.append(predict)
#             file_names.append(file_name)

#         submission = pd.DataFrame()
#         submission['PredictionString'] = prediction_strings
#         submission['image_id'] = file_names
#         submission.to_csv(osp.join(self.output_dir, 'submission.csv'), index=None)
#         print(f'submission saved to {osp.join(self.output_dir, "submission.csv")}')
