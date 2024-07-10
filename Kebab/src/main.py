from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Nombre del modelo en Hugging Face
model_name = "nombre_del_modelo/phi3"  # Reemplaza con el nombre correcto

# Descargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Guardar el modelo localmente
tokenizer.save_pretrained("./modelo_phi3")
model.save_pretrained("./modelo_phi3")

# Cargar el modelo y el tokenizador desde el almacenamiento local
tokenizer = AutoTokenizer.from_pretrained("./modelo_phi3")
model = AutoModelForCausalLM.from_pretrained("./modelo_phi3")
model.to("cpu")

@app.post("/predict/")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt")
    with torch.no_grad():  # Desactivar autograd para ahorrar memoria
        outputs = model.generate(inputs.input_ids)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
