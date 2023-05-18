# Databricks notebook source
# MAGIC %md
# MAGIC # Description
# MAGIC
# MAGIC  1. we read data from a table provided by Mo
# MAGIC  2. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Read the data and then create and save a small subset to parquet

# COMMAND ----------

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

# COMMAND ----------

table_mo="enterprise_azlab_prod.magratheav2"
# table_mo_sample = "enterprise_azlab_prod.magratheav2_0.001sample"


# COMMAND ----------

sdf_mo_table = spark.sql(f"select * from {table_mo}")

sdf_mo_table.cache()
print(sdf_mo_table.count())
sdf_mo_table.display()

# COMMAND ----------

sdf = sdf_mo_table
sdf.printSchema()

# COMMAND ----------

# sdf_sample = sdf.sample(False, fraction=0.001 )
# sdf_sample.cache()
# print(sdf_sample.count(), sdf.count())
# sdf_sample.coalesce(1).write.format("parquet").mode("overwrite").save("/mnt/enterprise/magrathea/magratheav2_0.001sample.parquet")

# COMMAND ----------

# sdf_sample = spark.read.parquet("/mnt/enterprise/magrathea/magratheav2_0.001sample.parquet")
# sdf_sample.cache()
# print(sdf_sample.count())
# sdf_sample.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Running dolly: a simple example

# COMMAND ----------

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)

# tokenizer = AutoTokenizer.from_pretrained("https://huggingface.co/databricks/dolly-v2-12b", padding_side="left")
# model = AutoModelForCausalLM.from_pretrained("https://huggingface.co/databricks/dolly-v2-12b", device_map="auto", trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", device_map="GPU", padding_side="left")
# model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="GPU", trust_remote_code=True)

# ValueError: If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'.

# COMMAND ----------

# DBTITLE 1,what is in the model?
model.named_parameters

# COMMAND ----------

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# # Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
# generate_response("Write a tweet announcing Dolly, a large language model from Databricks.", model=model, tokenizer=tokenizer)

# COMMAND ----------

2+10

# COMMAND ----------

1+1

# COMMAND ----------

# MAGIC %md #Run above

# COMMAND ----------

aPrompt ="""I am John, a 41 year old man from London. Last week I bought a pair of faded blue jeans, some black socks and a green jumper from Marks and Spencer.Today I am looking at a page on the Marks and Spencer website looking at shirts. What would you recommend John to buy from Marks and Spencer next?"""

# COMMAND ----------

generate_response(aPrompt, model=model, tokenizer=tokenizer)

# COMMAND ----------

aPrompt2 ="""I am John, a 41 year old man from London. Last week I bought a pair of faded blue jeans, some black socks and a green jumper from Marks and Spencer.Today I am looking at a page on the Marks and Spencer website looking at shirts. What color shirt would you recommend John to buy from Marks and Spencer next?"""

# COMMAND ----------

generate_response(aPrompt2, model=model, tokenizer=tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generating SQL from table schema

# COMMAND ----------

aPrompt2= """
The table magratheav2 has the following schema
 |-- sales_amt: decimal(28,2) (nullable = true)
 |-- item_qty: long (nullable = true)
 |-- storeName: string (nullable = true)
 |-- department: string (nullable = true)
 |-- storeNumber: string (nullable = true)
 |-- naturalDate: date (nullable = true)
 |-- trading_area: string (nullable = true)
 |-- department_name: string (nullable = true)
 |-- business_area: string (nullable = true)
 |-- productname: string (nullable = true)
 |-- format: string (nullable = true)
 |-- division_name: string (nullable = true)
 |-- region_name: string (nullable = true)

What is a sql query for selecting storeName with highest sales_amt?
"""

# COMMAND ----------

result = generate_response(aPrompt2, model=model, tokenizer=tokenizer)
print(result)

# COMMAND ----------

aPrompt3= """
The table magratheav2 has the following schema
 |-- sales_amt: decimal(28,2) (nullable = true)
 |-- item_qty: long (nullable = true)
 |-- storeName: string (nullable = true)
 |-- department: string (nullable = true)
 |-- storeNumber: string (nullable = true)
 |-- naturalDate: date (nullable = true)
 |-- trading_area: string (nullable = true)
 |-- department_name: string (nullable = true)
 |-- business_area: string (nullable = true)
 |-- productname: string (nullable = true)
 |-- format: string (nullable = true)
 |-- division_name: string (nullable = true)
 |-- region_name: string (nullable = true)

What is a sql query for selecting the name of a store with highest sales amount?
"""

# COMMAND ----------

generate_response(aPrompt3, model=model, tokenizer=tokenizer)

# COMMAND ----------

aPrompt4= """
The table magratheav2 has the following schema
 |-- sales_amt: decimal(28,2) (nullable = true)
 |-- item_qty: long (nullable = true)
 |-- storeName: string (nullable = true)
 |-- department: string (nullable = true)
 |-- storeNumber: string (nullable = true)
 |-- naturalDate: date (nullable = true)
 |-- trading_area: string (nullable = true)
 |-- department_name: string (nullable = true)
 |-- business_area: string (nullable = true)
 |-- productname: string (nullable = true)
 |-- format: string (nullable = true)
 |-- division_name: string (nullable = true)
 |-- region_name: string (nullable = true)

What is a sql query for selecting the name of a store with highest sales?
"""

# COMMAND ----------

result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
print(result)

# COMMAND ----------

# delete
# spark.sql("SELECT storeName FROM magratheav2 GROUP BY storeName ORDER BY SUM(sales_amt) DESC").display()

# COMMAND ----------

# MAGIC %md
# MAGIC # From a question to result

# COMMAND ----------

prompt_question="""
What is the name of the store with highest sales?
"""
prompt_shcema= """
The table magratheav2 has the following schema
 |-- sales_amt: decimal(28,2) (nullable = true)
 |-- item_qty: long (nullable = true)
 |-- storeName: string (nullable = true)
 |-- department: string (nullable = true)
 |-- storeNumber: string (nullable = true)
 |-- naturalDate: date (nullable = true)
 |-- trading_area: string (nullable = true)
 |-- department_name: string (nullable = true)
 |-- business_area: string (nullable = true)
 |-- productname: string (nullable = true)
 |-- format: string (nullable = true)
 |-- division_name: string (nullable = true)
 |-- region_name: string (nullable = true)
"""

prompt_additional_data="""
The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
"""

prompt_sql=f"""
What is a sql query for {prompt_question}
"""

aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
print(aPrompt4)

# COMMAND ----------

result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
print(result)

# COMMAND ----------

  prompt_shcema= """
  The table magratheav2 has the following schema
  |-- sales_amt: decimal(28,2) (nullable = true)
  |-- item_qty: long (nullable = true)
  |-- storeName: string (nullable = true)
  |-- department: string (nullable = true)
  |-- storeNumber: string (nullable = true)
  |-- naturalDate: date (nullable = true)
  |-- trading_area: string (nullable = true)
  |-- department_name: string (nullable = true)
  |-- business_area: string (nullable = true)
  |-- productname: string (nullable = true)
  |-- format: string (nullable = true)
  |-- division_name: string (nullable = true)
  |-- region_name: string (nullable = true)
  """

# COMMAND ----------

def sql_generator():
  prompt_question="""
  What is the name of the store with highest sales?
  """
  prompt_shcema= """
  The table magratheav2 has the following schema
  |-- sales_amt: decimal(28,2) (nullable = true)
  |-- item_qty: long (nullable = true)
  |-- storeName: string (nullable = true)
  |-- department: string (nullable = true)
  |-- storeNumber: string (nullable = true)
  |-- naturalDate: date (nullable = true)
  |-- trading_area: string (nullable = true)
  |-- department_name: string (nullable = true)
  |-- business_area: string (nullable = true)
  |-- productname: string (nullable = true)
  |-- format: string (nullable = true)
  |-- division_name: string (nullable = true)
  |-- region_name: string (nullable = true)
  """

  prompt_additional_data="""
  The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
  """

  prompt_sql=f"""
  What is a sql query for {prompt_question}
  """

  aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
  print(aPrompt4)

# COMMAND ----------


def clean_sql_string2(sql_string_input, substring="SELECT"):
    index = sql_string_input.find(substring)
    if index != -1:
        return sql_string_input[index:]
    else:
        return ""


print(clean_sql_string2(result))

# COMMAND ----------


sdf = spark.sql(clean_sql_string2(result))
output_dict =sdf.toPandas().to_dict()
output_dict

# COMMAND ----------

result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
print(prompt_question,  result2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Quick Witted v0

# COMMAND ----------

def qw0(prompt_question, printing=False):

    prompt_shcema= """
    The table enterprise_azlab_prod.magratheav2 has the following schema
    |-- sales_amt: decimal(28,2) (nullable = true)
    |-- item_qty: long (nullable = true)
    |-- storeName: string (nullable = true)
    |-- department: string (nullable = true)
    |-- storeNumber: string (nullable = true)
    |-- naturalDate: date (nullable = true)
    |-- trading_area: string (nullable = true)
    |-- department_name: string (nullable = true)
    |-- business_area: string (nullable = true)
    |-- productname: string (nullable = true)
    |-- format: string (nullable = true)
    |-- division_name: string (nullable = true)
    |-- region_name: string (nullable = true)
    """

    # prompt_additional_data="""
    # The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
    # """
    prompt_additional_data="""
    """

    prompt_sql=f"""
    What is a sql query for {prompt_question}
    """

    aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
    # if printing:
    #   print(aPrompt4)

    result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
    # if printing:
    #   print(result)

    def clean_sql_string2(sql_string_input, substring="SELECT"):
        index = sql_string_input.find(substring)
        if index != -1:
            return sql_string_input[index:]
        else:
            return ""


    if printing:
      print(clean_sql_string2(result))

    sdf = spark.sql(clean_sql_string2(result))
    output_dict =sdf.toPandas().to_dict()
    # if printing:
    #   print("result in dict:" , output_dict)
    prompt_result =f"""We have the result {output_dict} from our computation."""
    aPrompt6 = prompt_result + prompt_question
    
    result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
    if printing: 
      print(prompt_question,  result2)
    
    return result2

# COMMAND ----------

def sql_creator(prompt_question, printing=False):

    prompt_shcema= """
    The table enterprise_azlab_prod.magratheav2 has the following schema
    |-- sales_amt: decimal(28,2) (nullable = true)
    |-- item_qty: long (nullable = true)
    |-- storeName: string (nullable = true)
    |-- department: string (nullable = true)
    |-- storeNumber: string (nullable = true)
    |-- naturalDate: date (nullable = true)
    |-- trading_area: string (nullable = true)
    |-- department_name: string (nullable = true)
    |-- business_area: string (nullable = true)
    |-- productname: string (nullable = true)
    |-- format: string (nullable = true)
    |-- division_name: string (nullable = true)
    |-- region_name: string (nullable = true)
    """

    # prompt_additional_data="""
    # The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
    # """
    prompt_additional_data="""
    """

    prompt_sql=f"""
    What is a sql query for {prompt_question}
    """

    aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
    # if printing:
    #   print(aPrompt4)

    result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
    # if printing:
    #   print(result)

    def clean_sql_string2(sql_string_input, substring="SELECT"):
        index = sql_string_input.find(substring)
        if index != -1:
            return sql_string_input[index:]
        else:
            return ""

    return clean_sql_string2(result), result

# COMMAND ----------

def qw1(prompt_question, printing=False):
    def clean_sql_string2(sql_string_input, substring="SELECT"):
        index = sql_string_input.find(substring)
        if index != -1:
            return sql_string_input[index:]
        else:
            return ""
    
    sql_clause, raw = sql_creator(prompt_question,True)

    sdf = spark.sql(clean_sql_string2(sql_clause))
    output_dict =sdf.toPandas().to_dict()
    # if printing:
    #   print("result in dict:" , output_dict)
    prompt_result =f"""We have the result {output_dict} from our computation."""
    aPrompt6 = prompt_result + prompt_question
    
    result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
    if printing: 
      print(prompt_question,  result2)
    
    return result2

# COMMAND ----------

def qw2(prompt_question, printing=False):
  def clean_sql_string2(sql_string_input, substring="SELECT"):
      index = sql_string_input.find(substring)
      if index != -1:
          return sql_string_input[index:]
      else:
          return ""
  
  sql_clause, raw = sql_creator(prompt_question,True)

  sdf = spark.sql(clean_sql_string2(sql_clause))
  output_dict =sdf.toPandas().to_dict()
  # if printing:
  #   print("result in dict:" , output_dict)
  prompt_result =f"""We have the result {output_dict} from our computation."""
  aPrompt6 = prompt_result + prompt_question
  aPrompt6= f"""{prompt_result} \
    {prompt_question}. \
      Provide the answer in JSON format with the following keys: 
area_highest_sales, total_sales.
  """
  
  result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
  if printing: 
    print(prompt_question,  result2)
  
  return result2

# COMMAND ----------

def qw3(prompt_question, printing=False):
    def clean_sql_string2(sql_string_input, substring="SELECT"):
        index = sql_string_input.find(substring)
        if index != -1:
            return sql_string_input[index:]
        else:
            return ""
    
    sql_clause, raw = sql_creator(prompt_question,True)
    
    sql_clean = clean_sql_string2(sql_clause)
    if printing:
        print(f"SQL: {sql_clean}")
    
    sdf = spark.sql(sql_clean)
    # sdf = spark.sql(clean_sql_string2(sql_clause))
    output_dict =sdf.toPandas().to_dict()
    # if printing:
    #   print("result in dict:" , output_dict)
    prompt_result =f"""We have the result {output_dict} from our computation."""
    aPrompt6 = prompt_result + prompt_question
    aPrompt6= f"""{prompt_result} \
      {prompt_question}. \
        Provide the answer in JSON format with the following keys: 
        store_name, area_highest_sales, total_sales.
    """
    
    result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
    if printing: 
      print(prompt_question,  result2)
    
    return result2

# COMMAND ----------

# prompt_question="""
# What is the name of the 3 stores with the highest sales?
# """
# result = qw0(prompt_question, printing=False)
# print(result )

# COMMAND ----------

prompt_question="""
What is the name of the store with the highest sales?
"""

sql_clause, raw = sql_creator(prompt_question,True)
print("raw: ", raw)
print("sql_clause: ",sql_clause )

# COMMAND ----------

prompt_question="""
What is the name of the 3 stores with the highest sales?
"""
result = qw0(prompt_question, printing=False)
print(result )

# COMMAND ----------

prompt_question="""
What is the name of the 3 store with highest sales?
"""
print(qw2(prompt_question,True))


# COMMAND ----------

prompt_question="""
What is the name of the store with second highest sales?
"""
print(qw2(prompt_question,True))


# COMMAND ----------

prompt_question="""
What is the name of the store with the highest sales?
"""
print(qw2(prompt_question,True))


# COMMAND ----------

prompt_question="""
What is the name of the store with second highest sales?
"""

sql_clause, raw = sql_creator(prompt_question,True)
print("raw: ", raw)
print("sql_clause: ",sql_clause )


# COMMAND ----------

prompt_question="""
What is the name of the store with second highest sales?
"""
print(qw3(prompt_question,True))


# COMMAND ----------

prompt_question="""
What is the name of the store with the highest sales?
"""
print(qw3(prompt_question,True))


# COMMAND ----------

prompt_question="""
What is the name of the store with the highest total sales amount?
"""
print(qw3(prompt_question,True))


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT item_qty, storeName, storeNumber, naturalDate, trading_area, department_name, business_area, productname, format, division_name, region_name, sales_amt 
# MAGIC FROM enterprise_azlab_prod.magratheav2 
# MAGIC WHERE sales_amt > (SELECT TOP(sales_amt) FROM enterprise_azlab_prod.magratheav2 ORDER BY sales_amt DESC) 
# MAGIC ORDER BY sales_amt DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT  
# MAGIC   max(sales_amt) sales_amt,
# MAGIC   item_qty,
# MAGIC   storeName,
# MAGIC   department,
# MAGIC   storeNumber,
# MAGIC   naturalDate,
# MAGIC   trading_area,
# MAGIC   department_name,
# MAGIC   business_area,
# MAGIC   productname,
# MAGIC   division_name,
# MAGIC   region_name
# MAGIC FROM enterprise_azlab_prod.magratheav2
# MAGIC group by storeName, department, storeNumber, naturalDate, trading_area, department_name, business_area, productname, division_name, region_name
# MAGIC order by 2 desc  -- sales_amt desc
# MAGIC limit 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing qw0

# COMMAND ----------

prompt_question="""
What is the name of the store with the second highest sales?
"""
print(qw0(prompt_question,True))

# COMMAND ----------

# MAGIC %md
# MAGIC # Quick Witted v1

# COMMAND ----------

def sql_creator(prompt_question, printing=False):

    prompt_shcema= """
    The table enterprise_azlab_prod.magratheav2 has the following schema
    |-- sales_amt: decimal(28,2) (nullable = true)
    |-- item_qty: long (nullable = true)
    |-- storeName: string (nullable = true)
    |-- department: string (nullable = true)
    |-- storeNumber: string (nullable = true)
    |-- naturalDate: date (nullable = true)
    |-- trading_area: string (nullable = true)
    |-- department_name: string (nullable = true)
    |-- business_area: string (nullable = true)
    |-- productname: string (nullable = true)
    |-- format: string (nullable = true)
    |-- division_name: string (nullable = true)
    |-- region_name: string (nullable = true)
    """

    # prompt_additional_data="""
    # The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
    # """
    prompt_additional_data="""
    """

    prompt_sql=f"""
    What is a sql query for {prompt_question}
    """

    aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
    # if printing:
    #   print(aPrompt4)

    result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
    # if printing:
    #   print(result)

    def clean_sql_string2(sql_string_input, substring="SELECT"):
        index = sql_string_input.find(substring)
        if index != -1:
            return sql_string_input[index:]
        else:
            return ""

    return clean_sql_string2(result), result

# COMMAND ----------

prompt_question="""
What is the department with the highest sales?
"""

sql_clause, raw = sql_creator(prompt_question,True)
print("raw: ", raw)
print("sql_clause: ",sql_clause )

# COMMAND ----------

def qw1(prompt_question, printing=False):
  def clean_sql_string2(sql_string_input, substring="SELECT"):
      index = sql_string_input.find(substring)
      if index != -1:
          return sql_string_input[index:]
      else:
          return ""
  
  sql_clause, raw = sql_creator(prompt_question,True)

  sdf = spark.sql(clean_sql_string2(sql_clause))
  output_dict =sdf.toPandas().to_dict()
  # if printing:
  #   print("result in dict:" , output_dict)
  prompt_result =f"""We have the result {output_dict} from our computation."""
  aPrompt6 = prompt_result + prompt_question
  
  result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
  if printing: 
    print(prompt_question,  result2)
  
  return result2

# COMMAND ----------



# COMMAND ----------

# # def qw1(prompt_question, printing=False):

#   prompt_shcema= """
#   The table enterprise_azlab_prod.magratheav2 has the following schema
#   |-- sales_amt: decimal(28,2) (nullable = true)
#   |-- item_qty: long (nullable = true)
#   |-- storeName: string (nullable = true)
#   |-- department: string (nullable = true)
#   |-- storeNumber: string (nullable = true)
#   |-- naturalDate: date (nullable = true)
#   |-- trading_area: string (nullable = true)
#   |-- department_name: string (nullable = true)
#   |-- business_area: string (nullable = true)
#   |-- productname: string (nullable = true)
#   |-- format: string (nullable = true)
#   |-- division_name: string (nullable = true)
#   |-- region_name: string (nullable = true)
#   """

#   # prompt_additional_data="""
#   # The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
#   # """
#   prompt_additional_data="""
#   """

#   prompt_sql=f"""
#   What is a sql query for {prompt_question}
#   """

#   aPrompt4 = prompt_shcema + prompt_additional_data+ prompt_sql
#   # if printing:
#   #   print(aPrompt4)

#   result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
#   # if printing:
#   #   print(result)

#   def clean_sql_string2(sql_string_input, substring="SELECT"):
#       index = sql_string_input.find(substring)
#       if index != -1:
#           return sql_string_input[index:]
#       else:
#           return ""


#   if printing:
#     print(clean_sql_string2(result))

#   sdf = spark.sql(clean_sql_string2(result))
#   output_dict =sdf.toPandas().to_dict()
#   # if printing:
#   #   print("result in dict:" , output_dict)
#   prompt_result =f"""We have the result {output_dict} from our computation."""
#   aPrompt6 = prompt_result + prompt_question
  
#   result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
#   if printing: 
#     print(prompt_question,  result2)
  
#   return result2

# COMMAND ----------

1+1

# COMMAND ----------

def qw2(prompt_question, printing=False):
  def clean_sql_string2(sql_string_input, substring="SELECT"):
      index = sql_string_input.find(substring)
      if index != -1:
          return sql_string_input[index:]
      else:
          return ""
  
  sql_clause, raw = sql_creator(prompt_question,True)

  sdf = spark.sql(clean_sql_string2(sql_clause))
  output_dict =sdf.toPandas().to_dict()
  # if printing:
  #   print("result in dict:" , output_dict)
  prompt_result =f"""We have the result {output_dict} from our computation."""
  aPrompt6 = prompt_result + prompt_question
  aPrompt6= f"""{prompt_result} \
    {prompt_question}. \
      Provide the answer in JSON format with the following keys: 
area_highest_sales, total_sales.
  """
  
  result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
  if printing: 
    print(prompt_question,  result2)
  
  return result2

# COMMAND ----------

prompt_question="""
What business_area has the highest sales?
"""
print(qw2(prompt_question,True))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


