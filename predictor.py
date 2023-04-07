import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

def predict(image_file):

    #load model with params
    model = models.efficientnet_b0(weights=None)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')), strict=False)
    device = torch.device('cpu')

    class_names = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

    def pred_image(image_path, model):
        topk = 3 

        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        img_normalized = transform(image).unsqueeze(0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_normalized = img_normalized.to(device)

        with torch.no_grad():
            model.eval()  
            output = model(img_normalized)
            probs, indices = torch.topk(torch.softmax(output, dim=1), topk)
            # index = output.data.cpu().numpy().argmax()
        tmp_lst = []
        print(indices)
        print(probs)
        for j in range(topk):
            tmp_dct = {}
            label_indx = indices[0][j]
            # print(label_indx)
            class_name = class_names[label_indx]
            tmp_dct["predicted"] =  class_name
            tmp_dct["probability"] =  probs[0][j]
            tmp_lst.append(tmp_dct)
                
                # print(f"Prediction {j+1}: label index: {indices[i][j]}, probability: {probs[i][j]:.4f}")

        # class_name = class_names[index]
        return tmp_lst

    predicted_label = pred_image(image_file,model)
    return predicted_label
