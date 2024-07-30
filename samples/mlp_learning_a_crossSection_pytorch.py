#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import commentjson as json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
import torch
import time
import SimpleITK as sitk
from PIL import Image as PIL_Image

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as si

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class Image(torch.nn.Module):
	def __init__(self, filename, device, scale):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)
		self.scale = scale

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()

			## downsample using scale
			indices = indices - indices % self.scale

			lerp_weights = (xs - indices.float()) / self.scale

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)

			## find next corner with correct scale.
			x1 = (x0 + self.scale).clamp(max=shape[1]-1)
			y1 = (y0 + self.scale).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

class CrossSection(torch.nn.Module):
	def __init__(self, filename, device, scale):
		super(CrossSection, self).__init__()
		self.data = self.load_file(filename)
		self.shape = self.data.shape
		print(f"shape={self.shape}")
		self.data = self.data.to(device)
		self.scale = scale
		print(f"self.scale={self.scale}")

	## load file into 3D tensor
	## TODO 
	## check if need to do normalization for each image to improve performance. 
	def load_file(self, filename):
		print(f"load_file={filename}")
		data = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(float)
		data = torch.from_numpy(data).float()[... , :10] ## TODO out of memory because too many images
		print(f"load_file_max={data.max()}")
		print(f"load_file_min={data.min()}")
		print(f"load_file={data.shape}")

		## normalize data
		data = data.permute(2, 1, 0)[..., None]
		data_min = data.min()
		data_max = data.max()
		data = (data - data_min) / (data_max - data_min)

		print(f"load_file_max={data.max()}")
		print(f"load_file_min={data.min()}")
		print(f"load_file={data.shape}")

		return data

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor(shape[:3], device=xs.device).float()
			indices = xs.long()

			## downsample using scale
			indices = indices - indices % self.scale

			lerp_weights = (xs - indices.float()) / self.scale

			x0 = indices[:, 2].clamp(min=0, max=shape[2]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[1]-1)
			z0 = indices[:, 0].clamp(min=0, max=shape[0]-1)

			## find next corner with correct scale.
			x1 = (x0 + self.scale).clamp(max=shape[2]-1)
			y1 = (y0 + self.scale).clamp(max=shape[1]-1)
			z1 = (z0 + self.scale).clamp(max=shape[0]-1)

			## TODO
			## need to implement 3D bilinear interpolation
			# return (
			# 	self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
			# 	self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
			# 	self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
			# 	self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			# )
			return (
				self.data[z0, y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) * (1.0 - lerp_weights[:,2:3]) +
				self.data[z0, y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) * (1.0 - lerp_weights[:,2:3]) +
				self.data[z0, y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] * (1.0 - lerp_weights[:,2:3]) +
				self.data[z0, y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2] * (1.0 - lerp_weights[:,2:3]) +
				self.data[z1, y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) * lerp_weights[:,2:3] +
				self.data[z1, y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) * lerp_weights[:,2:3] +
				self.data[z1, y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] * lerp_weights[:,2:3] +
				self.data[z1, y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2] * lerp_weights[:,2:3]
			)

## takes in numpy array of images of 4D
def write_3d_image(path, imgs):
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(imgs):
        # print(f"img.max={img.max()} img.min={img.min()}")
        img = np.squeeze(img, axis=-1)
        img = np.uint8(img * 255.)
        # print(f"img={img.shape}")
        PIL_Image.fromarray(img).save(os.path.join(path, str(i).zfill(3) + '.png'))


def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("--scale", nargs="?", type=int, default=1, help="Scale factor for the image")
	parser.add_argument("--image", nargs="?", default="/home/simtech/Qiming/kits19/data/case_00150/case_00150.nii.gz", help="Image to match")
	parser.add_argument("--config", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("--n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
	parser.add_argument("--result_filename", nargs="?", default="", help="Number of training steps")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	print("================================================================")
	print("This script replicates the behavior of the native CUDA example  ")
	print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
	print("================================================================")

	print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

	device = torch.device("cuda")
	args = get_args()

	with open(args.config) as config_file:
		config = json.load(config_file)

	cross_section = CrossSection(args.image, device, args.scale)
	n_channels = cross_section.data.shape[3] ## assume cross section 3D data + gray scale
	cs = cross_section.data.cpu().numpy()

	model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	print(model)

	#===================================================================================================
	# The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
	# tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
	#===================================================================================================
	# encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
	# network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
	# model = torch.nn.Sequential(encoding, network)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Variables for saving/displaying image results
	resolution = cross_section.data.shape[0:3]
	img_shape = cross_section.data.shape ## here is 4D
	n_pixels = resolution[0] * resolution[1] * resolution[2]

	half_dx =  0.5 / resolution[2]
	half_dy =  0.5 / resolution[1]
	half_dz =  0.5 / resolution[0]
	xs = torch.linspace(half_dx, 1-half_dx, resolution[2], device=device)
	ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
	zs = torch.linspace(half_dz, 1-half_dz, resolution[0], device=device)
	zv, yv, xv = torch.meshgrid([zs, ys, xs])

	zyx = torch.stack((zv.flatten(), yv.flatten(), xv.flatten()), dim=-1)

	print(f"zyx={zyx.shape}")

	path = f"reference/"
	print(f"Writing '{path}'... ", end="")
	gt = cross_section(zyx).reshape(img_shape).detach().cpu().numpy()
	write_3d_image(path, gt)
	print("done.")

	prev_time = time.perf_counter()

	batch_size = 2**18
	interval = 10

	print(f"Beginning optimization with {args.n_steps} training steps.")

	## TODO
	## add this one back for performance.
	try:
		batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
		traced_image = torch.jit.trace(cross_section, batch)
	except:
		# If tracing causes an error, fall back to regular execution
		print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
		traced_image = cross_section

	# traced_image = image

	for i in range(args.n_steps):
		batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
		targets = traced_image(batch)
		output = model(batch)

		relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
		loss = relative_l2_error.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % interval == 0:
			loss_val = loss.item()
			torch.cuda.synchronize()
			elapsed_time = time.perf_counter() - prev_time
			print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

			path = f"save/{i}"
			print(f"Writing '{path}'... ", end="")
			with torch.no_grad():
				pred = model(zyx).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
				write_3d_image(path, pred)
				# print(f"pred={pred.shape} gt={gt.shape}")
				# print(f"psnr={psnr(img, pred, data_range=1)} ssim={si(img, pred, channel_axis=-1)}")
				## find average psnr and ssim
				psnr_val = 0
				ssim_val = 0
				for i in range(pred.shape[0]):
					psnr_val += psnr(gt[i], pred[i], data_range=1)
					ssim_val += si(gt[i], pred[i], channel_axis=-1)
				psnr_val /= pred.shape[0]
				ssim_val /= pred.shape[0]
				print(f"psnr={psnr_val} ssim={ssim_val}")
			print("done.")

			# Ignore the time spent saving the image
			prev_time = time.perf_counter()

			if i > 0 and interval < 1000:
				interval *= 10

	# if args.result_filename:
	# 	print(f"Writing '{args.result_filename}'... ", end="")
	# 	with torch.no_grad():
	# 		write_image(args.result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
	# 	print("done.")

	tcnn.free_temporary_memory()
