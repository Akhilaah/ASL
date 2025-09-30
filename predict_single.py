import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type:ignore

model = load_model("asl_model.h5")
class_indices = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J",
                 10:"K",11:"L",12:"M",13:"N",14:"O",15:"P",16:"Q",17:"R",18:"S",19:"T",
                 20:"U",21:"V",22:"W",23:"X",24:"Y",25:"Z",26:"del",27:"nothing",28:"space"}

img_path = "test_images/A_test.jpg"  # change image
img = image.load_img(img_path, target_size=(64,64))
img_array = np.expand_dims(image.img_to_array(img), axis=0)/255.0
pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)[0]

print(f"{img_path} → {class_indices[predicted_class]}")
