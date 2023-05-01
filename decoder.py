import torch.nn as nn
import torch
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
from wama_modules.thirdparty_lib.MedicalNet_Tencent.model import generate_model
from wama_modules.Encoder import ResNetEncoder,ResNet18Encoder
from wama_modules.Transformer import TransformerEncoderLayer, TransformerDecoderLayer
from wama_modules.BaseModule import MakeNorm, GlobalMaxPool
from wama_modules.Decoder import UNet_decoder
from wama_modules.Head import SegmentationHead,ClassificationHead
from wama_modules.utils import resizeTensor
from wama_modules.thirdparty_lib.C3D_jfzhang95.c3d import C3D
from wama_modules.PositionEmbedding import PositionalEncoding_1D_learnable
import torch.nn.functional as F
norm_set = 'gn'
gn_c = 8


class Multi_task_Unet(nn.Module):
		def __init__(self,
					 in_channel,
					 seg_label_category_dict,
					 cls_label_category_dict,
					 dim=2):
			super().__init__()
			# encoder
			Encoder_f_channel_list = [64, 128, 256, 512]
			self.encoder = ResNetEncoder(
				in_channel,
				stage_output_channels=Encoder_f_channel_list,
				stage_middle_channels=Encoder_f_channel_list,
				blocks=[1, 2, 3, 4],
				type='131',
				downsample_ration=[0.5, 0.5, 0.5, 0.5],
				dim=dim)

			# decoder
			Decoder_f_channel_list = [32, 64, 128]
			self.decoder = UNet_decoder(
				in_channels_list=Encoder_f_channel_list,
				skip_connection=[False, True, True],
				out_channels_list=Decoder_f_channel_list,
				dim=dim)
			# seg head
			self.seg_head = SegmentationHead(
				seg_label_category_dict,
				Decoder_f_channel_list[0],
				dim=dim)
			# cls head
			self.cls_head = ClassificationHead(cls_label_category_dict, Encoder_f_channel_list[-1])
			# regression head
			self.reg_head1 = nn.Linear(512*2*3*3,1130)
			# pooling
			self.pooling = GlobalMaxPool()

		def forward(self, x):
			# get encoder features
			multi_scale_encoder = self.encoder(x)
			# get decoder features
			multi_scale_decoder = self.decoder(multi_scale_encoder)
			# perform segmentation
			f_for_seg = resizeTensor(multi_scale_decoder[0], size=x.shape[2:])
			seg_logits = self.seg_head(f_for_seg)
			# perform classification
			cls_logits = self.cls_head(self.pooling(multi_scale_encoder[-1]))
			#perform regression
			reg_logits = self.reg_head1(multi_scale_encoder[-1].view(multi_scale_encoder[-1].shape[0],-1))
			return seg_logits, cls_logits, reg_logits

if __name__ == '__main__':
	x = torch.ones([3, 1, 96, 128, 128])
	seg_category_dict = dict(seg_roi=1,feature_map=1)
	cla_category_dict = dict(tumor=2)
	model = Multi_task_Unet(in_channel=1, seg_label_category_dict=seg_category_dict,cls_label_category_dict=cla_category_dict, dim=3)
	seg_logits,cla_logits,reg_logits = model(x)
	print('multi-label predicted logits')
	_ = [print('logits of ', key, ':', seg_logits[key].shape) for key in seg_logits.keys()]
	_ = [print('logits of ', key, ':', cla_logits[key].shape) for key in cla_logits.keys()]
	_ = [print('logits of regression', ':', reg_logits.shape) ]









