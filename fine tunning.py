!pip uninstall -y torchao
!pip install torchao==0.16.0
import torch
from transformers import ( AutoTokenizer,AutoModelForCausalLM,
    Trainer,TrainingArguments,DataCollatorForLanguageModeling
)
from peft import LoraConfig,get_peft_model
import pandas as pd
from datasets import load_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


model = AutoModelForCausalLM.from_pretrained(
    model_name
)


data = {
    "questions" :[
    "What are your store timings?",
    "Is the supermarket open today?",
    "What time does the store close?",
    "Where is your nearest branch?",
    "Do you offer home delivery?",
    "Is same-day delivery available?",
    "How can I track my order?",
    "What are the delivery charges?",
    "Is free delivery available?",
    "Can I order groceries online?",
    "Do you have fresh vegetables?",
    "Are organic fruits available?",
    "Do you sell dairy products?",
    "Is fresh milk available today?",
    "Do you have brown eggs?",
    "Are frozen foods available?",
    "Do you sell bakery items?",
    "Is fresh bread available?",
    "Do you have birthday cakes?",
    "Do you sell soft drinks?",
    "Do you have snacks?",
    "Are chocolates available?",
    "Do you sell baby products?",
    "Are diapers available?",
    "Do you have baby formula?",
    "Do you sell cosmetics?",
    "Are skincare products available?",
    "Do you sell shampoo?",
    "Do you have soaps?",
    "Do you sell toothpaste?",
    "Are cleaning products available?",
    "Do you have floor cleaners?",
    "Do you sell detergents?",
    "Are kitchen utensils available?",
    "Do you sell storage containers?",
    "Do you have bottled water?",
    "Are cooking oils available?",
    "Do you sell rice?",
    "Is wheat flour available?",
    "Do you sell pulses?",
    "Do you have spices?",
    "Are dry fruits available?",
    "Do you sell tea and coffee?",
    "Do you have sugar-free products?",
    "Are gluten-free products available?",
    "Do you have vegan products?",
    "Are there any discounts today?",
    "What are today's special offers?",
    "Do you have combo deals?",
    "Can I use discount coupons?",
    "Do you accept credit cards?",
    "Can I pay using UPI?",
    "Do you accept cash payments?",
    "Is EMI available?",
    "Do you have a loyalty program?",
    "How can I earn reward points?",
    "Can I redeem reward points?",
    "How do I become a member?",
    "What is your return policy?",
    "Can I exchange a product?",
    "How many days do I have for returns?",
    "What if I receive a damaged product?",
    "Can I cancel my order?",
    "How do I request a refund?",
    "Do you offer gift cards?",
    "Can I buy gift vouchers?",
    "Is gift wrapping available?",
    "Do you sell pet food?",
    "Are pet accessories available?",
    "Do you have stationery items?",
    "Do you sell school supplies?",
    "Are toys available?",
    "Do you sell kitchen appliances?",
    "Do you have electronic accessories?",
    "Are mobile chargers available?",
    "Do you sell batteries?",
    "Can I check product availability online?",
    "Is this product in stock?",
    "When will the item be restocked?",
    "Can you reserve a product for me?",
    "Do you have parking facilities?",
    "Is wheelchair access available?",
    "Are shopping carts available?",
    "Do you provide reusable shopping bags?",
    "Can I bring my own shopping bag?",
    "Do you have self-checkout counters?",
    "How long is the billing queue?",
    "Is customer support available?",
    "How can I contact customer care?",
    "Do you have a complaint desk?",
    "Can I provide feedback online?",
    "Do you sell seasonal products?",
    "Do you have festival gift hampers?",
    "Do you offer bulk purchase discounts?",
    "Can businesses place wholesale orders?",
    "Do you provide GST invoices?",
    "Can I schedule my grocery delivery?",
    "Are fresh fruits available today?",
    "What are your best-selling products?",
    "What new products have arrived this week?",
    ],
"answers" : [
    "Our store is open daily from 9:00 AM to 9:00 PM.",
    "Yes, we are open today.",
    "The store closes at 9:00 PM.",
    "Please share your location to find the nearest branch.",
    "Yes, we offer home delivery.",
    "Yes, same-day delivery is available for eligible orders.",
    "You can track your order using the tracking link sent after purchase.",
    "Delivery charges depend on your location and order value.",
    "Yes, free delivery is available on eligible orders.",
    "Yes, groceries can be ordered online.",
    "Yes, fresh vegetables are available.",
    "Yes, organic fruits are available.",
    "Yes, we sell dairy products.",
    "Yes, fresh milk is available today.",
    "Yes, brown eggs are available.",
    "Yes, frozen foods are available.",
    "Yes, bakery items are available.",
    "Yes, fresh bread is available.",
    "Yes, birthday cakes are available.",
    "Yes, soft drinks are available.",
    "Yes, snacks are available.",
    "Yes, chocolates are available.",
    "Yes, baby products are available.",
    "Yes, diapers are available.",
    "Yes, baby formula is available.",
    "Yes, cosmetics are available.",
    "Yes, skincare products are available.",
    "Yes, shampoo is available.",
    "Yes, soaps are available.",
    "Yes, toothpaste is available.",
    "Yes, cleaning products are available.",
    "Yes, floor cleaners are available.",
    "Yes, detergents are available.",
    "Yes, kitchen utensils are available.",
    "Yes, storage containers are available.",
    "Yes, bottled water is available.",
    "Yes, cooking oils are available.",
    "Yes, different varieties of rice are available.",
    "Yes, wheat flour is available.",
    "Yes, pulses are available.",
    "Yes, spices are available.",
    "Yes, dry fruits are available.",
    "Yes, tea and coffee are available.",
    "Yes, sugar-free products are available.",
    "Yes, gluten-free products are available.",
    "Yes, vegan products are available.",
    "Yes, discounts are available on selected items.",
    "Today's special offers are available in-store and online.",
    "Yes, combo deals are available.",
    "Yes, valid discount coupons are accepted.",
    "Yes, we accept credit cards.",
    "Yes, UPI payments are accepted.",
    "Yes, cash payments are accepted.",
    "EMI is available on eligible purchases.",
    "Yes, we have a loyalty program.",
    "You earn reward points on eligible purchases.",
    "Yes, reward points can be redeemed.",
    "You can become a member by registering online or at the store.",
    "Products can be returned according to our return policy.",
    "Yes, eligible products can be exchanged.",
    "Returns are accepted within the return period.",
    "Damaged products can be replaced or refunded.",
    "Yes, orders can be canceled before dispatch.",
    "You can request a refund through customer support.",
    "Yes, gift cards are available.",
    "Yes, gift vouchers are available.",
    "Yes, gift wrapping is available.",
    "Yes, pet food is available.",
    "Yes, pet accessories are available.",
    "Yes, stationery items are available.",
    "Yes, school supplies are available.",
    "Yes, toys are available.",
    "Yes, kitchen appliances are available.",
    "Yes, electronic accessories are available.",
    "Yes, mobile chargers are available.",
    "Yes, batteries are available.",
    "Yes, product availability can be checked online.",
    "Please check the product page for stock availability.",
    "The restocking date depends on supplier availability.",
    "Yes, selected products can be reserved.",
    "Yes, parking facilities are available.",
    "Yes, wheelchair access is available.",
    "Yes, shopping carts are available.",
    "Yes, reusable shopping bags are available.",
    "Yes, you can bring your own shopping bag.",
    "Yes, self-checkout counters are available.",
    "Billing time depends on customer traffic.",
    "Yes, customer support is available.",
    "You can contact customer care by phone, email, or chat.",
    "Yes, a complaint desk is available.",
    "Yes, feedback can be submitted online.",
    "Yes, seasonal products are available.",
    "Yes, festival gift hampers are available.",
    "Yes, bulk purchase discounts are available.",
    "Yes, businesses can place wholesale orders.",
    "Yes, GST invoices are provided.",
    "Yes, you can schedule your grocery delivery.",
    "Yes, fresh fruits are available today.",
    "Fresh produce, dairy products, snacks, and beverages are our best-selling products.",
    "This week's new arrivals include seasonal groceries and household essentials."]}

df=pd.DataFrame(data)
df.to_json("train_data.json",orient="records",indent=4)
dataset=load_dataset("json",data_files="train_data.json")
print(dataset)
def formatting(data):
  return {"text":f"questions:{data['questions']} answers:{data['answers']}"}
dataset = dataset['train'].map(formatting)
print(dataset)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
def tokenize(prompt):
  result=tokenizer(prompt['text'],padding="max_length",truncation=True,max_length=128)
  result['labels']=result["input_ids"].copy()
  return result
dataset=dataset.map(tokenize)
print(dataset)
config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model=get_peft_model(model,config)
print(model.print_trainable_parameters)
data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
training_args=TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="steps",
    fp16=True
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator)
trainer.train()
model.save_pretrained("tinyllama_adapter")
tokenizer.save_pretrained("tinyllama_adapter")

device=next(model.parameters()).device
prompt="""
Question:
What are your store timings?
Answer:
"""
inputs=tokenizer(prompt,return_tensors="pt")
inputs={k:v.to(device) for k,v in inputs.items()}
output=model.generate(**inputs,max_new_tokens=100)
print(tokenizer.decode(output[0],skip_special_tokens=True))
