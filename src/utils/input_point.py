def get_best_mask_with_clip(image_path, input_point, input_label, text_prompt):
    # --- PHẦN 1: SAM SINH MASK ---
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image)
    
    # SAM trả về 3 masks (multimask_output=True)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([input_point]),
        point_labels=np.array([input_label]), # 1 là foreground
        multimask_output=True,
    )

    # --- PHẦN 2: CLIP CHẤM ĐIỂM ---
    best_score = -1.0
    best_mask_index = 0
    
    # Tokenize text prompt
    text_token = clip.tokenize([text_prompt]).to(device)

    # Duyệt qua từng mask mà SAM đề xuất
    for i, mask in enumerate(masks):
        # Tạo ảnh crop chỉ chứa vật thể trong mask
        # (Làm đen background để CLIP tập trung vào vật thể)
        masked_img = image.copy()
        masked_img[mask == False] = 0 # Che nền bằng màu đen
        
        # Crop ảnh theo bounding box của mask để loại bỏ vùng đen thừa
        y_indices, x_indices = np.where(mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        cropped_obj = masked_img[y_min:y_max, x_min:x_max]
        cropped_pil = Image.fromarray(cropped_obj)

        # Preprocess ảnh cho CLIP
        img_tensor = preprocess(cropped_pil).unsqueeze(0).to(device)

        # Tính toán similarity
        with torch.no_grad():
            image_features = clip_model.encode_image(img_tensor)
            text_features = clip_model.encode_text(text_token)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Tính Cosine Similarity
            similarity = (image_features @ text_features.T).item()
            
        print(f"Mask {i}: Similarity score = {similarity:.4f}")
        
        if similarity > best_score:
            best_score = similarity
            best_mask_index = i

    print(f"--> Chọn Mask số {best_mask_index} khớp nhất với mô tả '{text_prompt}'")
    return masks[best_mask_index]

# Ví dụ sử dụng:
# Người dùng click tại tọa độ (500, 300), prompt là "con mèo"
final_mask = get_best_mask_with_clip("anh_goc.jpg", [500, 300], [1], "a cat")