import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("./modelo_phi3")
model = AutoModelForCausalLM.from_pretrained("./modelo_phi3/")
model.to("cpu")

# Mensaje de entrada
messages = """
The project is in C#. Aitor is an expert in python, rust and nodejs. Should he review a PR about C#? Answer with 'yes' or 'no'.
"""

# Tokenizar la entrada y generar el attention mask
inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
attention_mask = inputs['attention_mask']

class StopOnYesNo(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length
    
    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][self.start_length:], skip_special_tokens=True)
        if "yes" in generated_text.lower() or "no" in generated_text.lower():
            return True
        return False

# Definir los criterios de parada
stop_criteria = StopOnYesNo(tokenizer, inputs.input_ids.shape[1])
stopping_criteria = StoppingCriteriaList([stop_criteria])

# Generar texto con parámetros controlados
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,  # Limitar la longitud de la respuesta
        num_return_sequences=1,  # Número de secuencias a devolver
        top_k=50,  # Limitar el número de tokens considerados durante la generación
        top_p=0.95,  # Probabilidad acumulada de los tokens considerados
        no_repeat_ngram_size=2,  # Evitar repeticiones de n-gramas
        early_stopping=True,  # Detener la generación temprano si se alcanza un token de finalización
        do_sample=True,  # Muestrear de la distribución de probabilidad de los tokens
        num_beams=2,  # Número de secuencias a considerar en cada paso
        attention_mask=inputs.attention_mask,
        stopping_criteria=stopping_criteria  # Criterios de parada personalizados
    )

# Decodificar la respuesta
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
# Filtrar y formatear la respuesta para solo devolver 'yes' o 'no'
if "yes" in response_text.split("### Answer:")[1].lower():
    response = "yes"
elif "no" in response_text.split("### Answer:")[1].lower():
    response = "no"
else:
    response = "No se pudo determinar una respuesta clara."

print(response)
