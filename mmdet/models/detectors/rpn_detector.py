import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .rpn import RPN


@DETECTORS.register_module()
class RPNDetector(RPN):

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        # Convert the rpn-proposals into bbox results format. <
        # proposal_list[0].shape = [200,5]
        bbox_results = []
        for det_bboxes in proposal_list:
            det_labels = torch.zeros((det_bboxes.size(0))).to(
                det_bboxes.device)
            bbox_results.append(
                bbox2result(det_bboxes, det_labels, num_classes=1))

        return bbox_results
        # >