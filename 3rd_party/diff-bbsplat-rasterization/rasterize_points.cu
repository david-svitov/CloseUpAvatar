/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\               
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

// return float pointer and sizes array
void textures_to_pointer(const std::vector<torch::Tensor>& tensor_list, const float** & data_ptrs, int* & d_sizes) {
    const int num_tensors = tensor_list.size();

    std::vector<const float*> host_data_ptrs(num_tensors);
    std::vector<int> host_sizes(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
        torch::Tensor tensor = tensor_list[i].contiguous();
        host_data_ptrs[i] = tensor.data_ptr<float>();
        host_sizes[i] = tensor.size(2);
    }

    cudaMalloc(&d_sizes, num_tensors * sizeof(int));
    cudaMemcpy(d_sizes, host_sizes.data(), num_tensors * sizeof(int), cudaMemcpyHostToDevice);

    const float** d_data_ptrs;
    cudaMalloc(&d_data_ptrs, num_tensors * sizeof(const float*));
    cudaMemcpy(d_data_ptrs, host_data_ptrs.data(), num_tensors * sizeof(const float*), cudaMemcpyHostToDevice);

    data_ptrs = d_data_ptrs;
}

void free_textures_pointers(const float** d_data_ptrs, int* d_sizes) {
    cudaFree(const_cast<float**>(d_data_ptrs));
    cudaFree(d_sizes);
}

void gradients_to_pointer(const std::vector<torch::Tensor>& tensor_list, float** & data_ptrs) {
    const int num_tensors = tensor_list.size();

    std::vector<float*> host_data_ptrs(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
        torch::Tensor tensor = tensor_list[i].contiguous();
        host_data_ptrs[i] = tensor.data_ptr<float>();
    }

    float** d_data_ptrs;
    cudaMalloc(&d_data_ptrs, num_tensors * sizeof(float*));
    cudaMemcpy(d_data_ptrs, host_data_ptrs.data(), num_tensors * sizeof(float*), cudaMemcpyHostToDevice);

    data_ptrs = d_data_ptrs;
}

void free_gradients_pointers(float** d_data_ptrs) {
    cudaFree(const_cast<float**>(d_data_ptrs));
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const std::vector<torch::Tensor>& texture_alpha,
	const std::vector<torch::Tensor>& texture_color,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (scales.ndimension() != 2 || scales.size(1) != 2) {
	AT_ERROR("scales must have dimensions (num_points, 2)");
  }

  if (rotations.ndimension() != 2 || rotations.size(1) != 4) {
	AT_ERROR("rotations must have dimensions (num_points, 4)");
  }

  int levels_number = texture_alpha.size();
  for (size_t i = 0; i < texture_alpha.size(); ++i) {
    const auto& texture = texture_alpha[i];
    if (texture.size(1) != texture.size(2)) {
      AT_ERROR("alpha texture should have square shape");
    }
    CHECK_INPUT(texture);
  }

  if (levels_number != texture_color.size()) {
    AT_ERROR("number of scales must be equal for alpha and rgb textures");
  }

  for (size_t i = 0; i < texture_color.size(); ++i) {
    const auto& texture = texture_color[i];
    if (texture.size(2) != texture.size(3)) {
      AT_ERROR("color texture should have square shape");
    }
    if (texture.size(1) != 3) {
      AT_ERROR("color texture should have 3 channels");
    }
    CHECK_INPUT(texture);
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+2, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor impact = torch::full({P}, 0.0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  // Convert lists of textures to float pointers
  const float** texture_alpha_ptr;
  int* texture_alpha_sizes;
  textures_to_pointer(texture_alpha, texture_alpha_ptr, texture_alpha_sizes);

  const float** texture_color_ptr;
  int* texture_color_size;
  textures_to_pointer(texture_color, texture_color_ptr, texture_color_size);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(),
		levels_number,
		texture_alpha_ptr,
		texture_alpha_sizes,
		texture_color_ptr,
		texture_color_size,
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		impact.contiguous().data<float>(),
		debug);
  }

  free_textures_pointers(texture_alpha_ptr, texture_alpha_sizes);
  free_textures_pointers(texture_color_ptr, texture_color_size);
  return std::make_tuple(rendered, out_color, out_others, radii, impact, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& out_colors,
	const torch::Tensor& out_others,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const std::vector<torch::Tensor>& texture_alpha,
	const std::vector<torch::Tensor>& texture_color,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  std::vector<torch::Tensor> dL_dtexture_alpha;
  std::vector<torch::Tensor> dL_dtexture_color;

  int levels_number = texture_alpha.size();
  const int num_textures = texture_alpha[0].size(0);
  for (size_t i = 0; i < texture_alpha.size(); ++i) {
    const auto& texture = texture_alpha[i];
    CHECK_INPUT(texture);

    // allocate memory for gradients
    const int texture_size = texture.size(1);
    torch::Tensor dL_dalpha = torch::zeros({num_textures, texture_size, texture_size}, means3D.options());
    dL_dtexture_alpha.push_back(dL_dalpha);
  }

  for (size_t i = 0; i < texture_color.size(); ++i) {
    const auto& texture = texture_color[i];
    CHECK_INPUT(texture);

    // allocate memory for gradients
    const int texture_size = texture.size(2);
    torch::Tensor dL_dcolor = torch::zeros({num_textures, 3, texture_size, texture_size}, means3D.options());
    dL_dtexture_color.push_back(dL_dcolor);
  }

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  // Convert lists of textures to float pointers
  const float** texture_alpha_ptr;
  int* texture_alpha_sizes;
  textures_to_pointer(texture_alpha, texture_alpha_ptr, texture_alpha_sizes);

  const float** texture_color_ptr;
  int* texture_color_size;
  textures_to_pointer(texture_color, texture_color_ptr, texture_color_size);

  // Get float pointers for gradients
  float** dL_dtexture_alpha_ptr;
  gradients_to_pointer(dL_dtexture_alpha, dL_dtexture_alpha_ptr);

  float** dL_dtexture_color_ptr;
  gradients_to_pointer(dL_dtexture_color, dL_dtexture_color_ptr);
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  levels_number,
	  texture_alpha_ptr,
	  texture_alpha_sizes,
	  texture_color_ptr,
	  texture_color_size,
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  out_colors.contiguous().data<float>(),
	  out_others.contiguous().data<float>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dtexture_alpha_ptr,
	  dL_dtexture_color_ptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  free_textures_pointers(texture_alpha_ptr, texture_alpha_sizes);
  free_textures_pointers(texture_color_ptr, texture_color_size);
  free_gradients_pointers(dL_dtexture_alpha_ptr);
  free_gradients_pointers(dL_dtexture_color_ptr);
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dtexture_alpha, dL_dtexture_color, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
