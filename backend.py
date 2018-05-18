from flask import Flask, jsonify, render_template, request
# from WordsAndPartsGenerating import GenerateSentenceWaP
# from WordsOnlyGenerating import GenerateSentenceWordOnly
# from LettersGenerating import GenerateLetters

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/txtGen/', methods=['GET'])
def txtGen():
    seedVal = request.args.get('seedVal')
    lengthVal = request.args.get('lengthVal')
    tempVal = request.args.get('tempVal')
    sourceVal = request.args.get('sourceVal')

    lengthVal = int(lengthVal)
    tempVal = int(tempVal)
    # resultWordsAndParts = GenerateSentenceWaP(seedVal, lengthVal, sourceVal)
    # resultOnlyWords = GenerateSentenceWordOnly(seedVal, lengthVal, tempVal, sourceVal)
    # resultLetters = GenerateLetters(seedVal, lengthVal, sourceVal)

    tmp = {
        'resultWoP': resultWordsAndParts,
        'resultWordOnly': resultOnlyWords,
        'resultLetters': resultLetters
    }

    return jsonify(tmp)


if __name__ == '__main__':
    app.run(port=8080, debug=True)