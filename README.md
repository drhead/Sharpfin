# Sharpfin
Library of custom Torchvision transforms built for accuracy, visual quality, and speed.

This library is being developed to address some common and often overlooked challenges in image data handling within the machine learning space, with a focus on making it easy to handle image data properly. It is currently in very early development (so early that I am not even guaranteeing anything in this repo works at all at this point), and implementations and even default settings may change frequently and without warning.

## Transforms library: `import sharpfin.transforms as SFT`

### SFT.Scale
For understanding why this module is necessary and helpful, let us first review what the most widely used options for image resize in the ML space are:

1. Pillow: Pillow has something very important that our other options don't: A working Lanczos implementation. Unfortunately, it also has a problem that most people overlook, in that it does not ensure that the transformation is applied in a linear color space. Pillow's CMS suite is excellent and relatively easy to use compared to some other options, so you can certainly work around this if you want to, but do you see anyone doing that in practice? Resize transformations assume linearity of the color space, and one will get incorrect results without this transformation.
2. PyTorch nn.functional.interpolate: Pytorch also has something very important that our other options don't: CUDA support. On a test benchmark, the highest quality available rescale for Pillow took about 12ms to complete, while the highest quality available F.interpolate rescale took 0.2ms. Unfortunately, that highest quality available rescale is bicubic with antialiasing. But there isn't anything fundamentally stopping us from doing a higher quality resize filter like Lanczos on a GPU.
3. OpenCV: OpenCV has infamously buggy rescale support, most notably the only remotely working downsampling method is INTER_AREA. It also lacks the same support for color management that Pillow has.

`SFT.Scale` aims to provide a better resizing transform that is primarily geared towards simplifying proper handling of image data. It has the following features:
- Better resizing kernels: In addition to having the Lanczos-2 and Lanczos-3 resizing kernels, we also support the Magic Kernel Sharp series of resizing kernels, which provide superior results to Lanczos-3. https://johncostella.com/magic/
- Comparable performance to existing options: On CUDA, our benchmark resize (random resize from the range of 950-1050 on each axis to 450-550) finished in 0.5ms using Magic Kernel Sharp 2021 versus 0.2ms for torch.nn.functional.interpolate with Bicubic+AA. This is with nothing but Python code and torch.compile, and a custom kernel could potentially bring this lower. Surprisingly, on CPU, our Magic Kernel Sharp 2021 performs better than the same kernel applied within PIL (15.5ms vs 17ms) despite working with full-precision floats and using an algorithm that is probably as FLOPS-inefficient as can be (but is nevertheless great for tensor cores, as the resizing work is a single matmul for each axis). Do note that using CUDA can potentially take away some resources from training.
- Flexible output options: For most ML dataloading applications, you will most likely be using the image in float format. `SFT.Scale` will do exactly that by default, so you will never have to worry about dithering. However, sometimes we want to save an image because we are preprocessing a dataset. For that, `SFT.Scale` will handle the quantization for you according to what you specify and return a `torch.uint8` tensor directly to minimize the risk of accidentally truncating floats when converting to a PIL image, for instance.
- Proper color space management. Image inputs (assumed to be sRGB) are converted to linear RGB, because interpolation transforms require a linear color space. You can disable this if your inputs are in another color space, but whatever color space you use will need to be linear or else it won't work well.

Note regarding use of GPU resources: A high-resolution image (6000x6000) would use about 200MB of memory in a 16 bit float format. The peak memory usage for `SFT.Scale` is a bit over twice the memory usage of the source image itself. You should account for this if using it in CUDA mode during a training run, for instance -- in many cases it is better and more than adequate to simply run resizing on the CPU unless your model is particularly high-throughput. CUDA can be much better for bulk preprocessing of images in advance of a training run.

#### **NEW** (4/10/2025): Sparsity Support
A custom Triton implementation of the resizing engine has now been added, making use of sparsity to avoid a sizeable portion of the matmul operation. This reduces the computational complexity for each resizing matmul from O(MNK) to O(NK). Sparsity code is adapted from stanford-futuredata/stk with changes mainly oriented at allowing more flexibility for input shapes. From my testing (on a fairly large operation -- 8192x8192 to 2050x2050), performance now seems to match F.interpolate on bicubic with antialiasing, and is 6x faster than the naive GPU implementation, and 1000x faster than the same operation on CPU using the naive implementation.

### SFT.ApplyCMS
`SFT.ApplyCMS` is intended to make proper color management of images as effortless as possible. Original implementation is largely made by @redhottensors.

PIL has great color management code, but it isn't used automatically. This module is intended to not only apply whatever ICC profile exists for an image (which usually would be done for you before display in a web browser or other application) before we convert it to a tensor and perform other operations, but also to properly handle certain aspects of the image such as transparency and rendering intent that are often overlooked as well.

It is packaged as a transform and should ideally be applied **immediately after loading an image** and **before all other transforms**. It will output a PIL image.

### SFT.AlphaComposite
`SFT.AlphaComposite` is recommended to go immediately after ApplyCMS, since ApplyCMS will output images with alpha channels if they exist. With this, you have control over what color the alpha channel is composited with. Default is pure white (255, 255, 255). Currently operates on PIL images only.

### SFT.AspectRatioCrop
`SFT.AspectRatioCrop` is intended to crop an image minimally to be the same aspect ratio as the provided resolution. If you don't use this before using `SFT.Scale`, it is very likely that your image will be distorted. (TODO: make this part of `SFT.Scale`, and leave the cropped area for use in place of edge padding where possible)

## Dataloader module: `import sharpfin.dataloader as SDL`

Sharpfin's dataloader module is in early development. The goal for it is to provide better preset dataloaders for common workloads, with a flexible and fast dataset specification.

### Dataset specification

Our dataset specification uses `msgspec`, because it is quite fast at loading data compared to virtually every other option. Msgspec typically uses Structs to deserialize data into. Our part of what is specified is barebones. For example, this is what an image dataset item looks like (skipping some internal bits):

```py
# Generic type variable for conditioning
T = TypeVar("T", bound=msgspec.Struct)

class ImageDatasetItem(msgspec.Struct, Generic[T]):
    image_path: str
    height: Optional[int] = None
    width: Optional[int] = None

    conditioning: Optional[T] = None
```

This doesn't seem like much. But you don't actually need to know more than this. You need to know where the image is, and ideally want to know *reliable* information about its dimensions (so that you can do things like aspect ratio bucketing done by `SDL.BucketedDataset`, which needs to be done in advance, without having to read every image first). The `conditioning` attribute is something that *you* will provide. You define your own `msgspec.Struct` with the conditioning information *your* model needs. We also provide an input for you to define your own function to do any necessary processing of your conditioning information, which will then be passed through and (hopefully) collated as part of the batch. Or, you can also just not use this. Go make a VAE, or an unconditional GAN for old times' sake. In any case, you can do that more easily with less code.
