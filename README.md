# Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## BERT4Rec

```bash
python main.py --template train_bert
```

## Examples

1. Train BERT4Rec on JSON and run test set inference after training

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```

# Test Set Results

Numbers under model names indicate the number of hidden layers.

