def prepare_mask_for_lama(binary_mask, dilate_kernel_size=15):
    # Chuyển mask sang uint8 (0 và 255)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    # Dilation (Nở mask)
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    return dilated_mask