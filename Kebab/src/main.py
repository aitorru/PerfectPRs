from fastapi import FastAPI
from llama_cpp import Llama

app = FastAPI()


llm = Llama(
  model_path="./phi3-gguf/Phi-3-mini-4k-instruct-fp16.gguf",  # path to GGUF file
  n_ctx=100,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=12, # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=50, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)

LANGS = ["Python", "Rust", "Node.js", "C#", "Java", "C++", "C", "Go", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "JavaScript", "HTML", "CSS", "SQL", "Shell", "PowerShell", "Objective-C", "Perl", "Scala", "Groovy", "Lua", "Dart", "Haskell", "Elixir", "Clojure", "Julia", "R", "Vim script", "Assembly", "Racket", "Erlang", "Crystal", "D", "Nim", "F#", "COBOL", "Pascal", "Dockerfile", "Makefile", "TeX", "VimL", "Emacs Lisp", "XSLT", "Roff", "Action"]

@app.post("/predict/fast/{lang}")
async def predict( lang: str ):
    prompt = f"The project is in {lang}. Aitor is an expert exclusively in Python, Rust, and Node.js. Should he review a PR about {lang}? Answer only with 'yes' or 'no'. If he should not review it, answer 'no'."
    output = llm(
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
        max_tokens=5,  # Generate up to 256 tokens
        stop=["<|end|>"], 
        echo=False,  # Whether to echo the prompt
    )
    return(output['choices'][0]['text'].split(".")[0])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
