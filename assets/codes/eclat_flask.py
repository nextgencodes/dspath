from flask import Flask, request, jsonify
import pandas as pd
from pyECLAT import ECLAT
from mlxtend.preprocessing import TransactionEncoder
import json

app = Flask(__name__)

# Dummy data and model (replace with actual data and model loading)
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk']
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
#frequent_itemsets = ECLAT(df, min_combination=0.5, use_colnames=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_data()
        data = pd.DataFrame(json.loads(data)['data'])
        # Run Eclat
        eclat = ECLAT(data=data)
        # the item shoud appear at least at 5% of transactions
        min_support =  0.5 # Default min_support is 0.5
        # start from transactions containing at least 2 items
        min_combination = 2
        # up to maximum items per transaction
        max_combination = 10
        rule_indices, rule_supports = eclat.fit(min_support=min_support,
                                                         min_combination=min_combination,
                                                         max_combination=max_combination,
                                                         separator=' & ',
                                                         verbose=True)

        result = pd.DataFrame(rule_supports.items(),columns=['Item', 'Support'])
        result.sort_values(by=['Support'], ascending=False)
        itemsets_dict = result.to_dict(orient='records')
        return jsonify({'result':itemsets_dict})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)