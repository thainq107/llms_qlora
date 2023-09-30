# Large Language Model Fine-Tuning with QLoRA
[QLoRA](https://github.com/artidoro/qlora): Efficient Fine-Tuning of QUantized LLMs

## Dependencies
- Python 3.10
- [PyTorch](https://github.com/pytorch/pytorch) 2.0 +
  ```
  pip install -r requirements.txt
  ```
## Dataset
  [carblacac/twitter-sentiment-analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
  
## Training
  ```
    python run_qlora.py \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --model_name_or_path tiiuae/falcon-7b \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```
