### Everything is explained in the good_luck.ipynb file.

graph LR
A[Input Image] --> B[LLaVA Vision Encoder]
B --> C[Projection Layer]
C --> D[Cerebras-GPT Decoder]
D --> E[Output CAD Code]

