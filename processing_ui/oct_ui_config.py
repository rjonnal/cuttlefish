c3_max = 1e-15
c3_min = -1e-15
c2_max = 1e-8
c2_min = -1e-8

c3_default = -9e-17

#c2_default = 13e-11 # worked for some of the 2019.11.27 2.0T data
c2_default = -48e-11

c3_power = -16
c2_power = -11

coef_single_step = 0.1
coef_significant_digits = 1

L1_default = 875e-9
L2_default = 825e-9

dc_crop_pixels = 50
bscan_linear_contrast_percentile = (5,99.5)
bscan_log_contrast_percentile = (50,99.5)

projection_z1_default = 134
projection_z2_default = 143

# When generating a TIFF, we have to decide how to
# discretize the floating point values in an OCT volume.
# It would be a mistake to use the full dynamic range of
# the 16-bit TIFF for every frame, because variations in
# the intensity of individual frames would be lost in
# process. Instead, have a fixed multiplier, specified here,
# and then round to an integer for 16-bit output. Anything
# over 2^16 would be clipped at 2^16, and anything below 0
# clipped at 0.
tiff_multiplier = 1.0

# off-axis filtering parameters; uses a 2D Gaussian filter
# centered about (offaxis_x0,offaxis_y0) with width parameter
# offaxis_sigma
offaxis_x0 = 283.5
offaxis_y0 = 101.5
offaxis_sigma = 167.0/3

dc_subtract_default = True
filter_volume_default = False
log_scale_default = False
image_extension = '.tif'
default_images_per_volume = 500
default_data_directory = '/home/rjonnal/Share/aossfoct/2019.11.27'
