#!flask/bin/python
import os
import datetime
import gpt_2_simple as gpt2

from flask import Flask
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

model_name = "124M"

if not os.path.isdir(os.path.join("models", model_name)):
    print("Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/


@app.route('/', methods=['GET'])
def start():
    print("Starting")
    start_time = datetime.datetime.now()

    sess = gpt2.start_tf_sess()

    gpt2.load_gpt2(sess, model_name=model_name)

    text = gpt2.generate(sess,
                         model_name=model_name,
                         prefix="In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
                                "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
                                "researchers was the fact that the unicorns spoke perfect English.",
                         length=100,
                         temperature=0.7,
                         top_p=0.9,
                         nsamples=1,
                         batch_size=1
                         )

    total_time = datetime.datetime.now() - start_time

    print("Total time required is = ", total_time)

    return text


if __name__ == '__main__':
    app.run(debug=True)
