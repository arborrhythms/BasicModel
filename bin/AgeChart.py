"""Externally observable presentations for the measured rows in ModelAge.md."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Question:
    text: str
    answer: str
    prompt: str | None = None
    context: tuple = ()


DATA = Path(__file__).resolve().parents[1] / 'data'


def measured_rows():
    return json.loads((DATA / 'lessons/age-chart.json').read_text())


def xor_mse_bar():
    return next(row['maximum_mse'] for row in measured_rows() if row['measure'] == 'exclusive or')


def chart():
    corpus = DATA / 'corpus-lessons.json'
    stages = json.loads(corpus.read_text())['stages']
    lesson = stages[0]
    sentences = tuple(sentence for document in lesson['documents']
        for record in document['records'] for sentence in record['sentences'])
    four_word_sentences = tuple(dict.fromkeys(sentence for stage in stages
        for document in stage['documents'] for record in document['records']
        for sentence in record['sentences'] if len(sentence.split()) == 4))
    words = tuple(dict.fromkeys(word for sentence in sentences for word in sentence[:-1].split()))
    counting = tuple(str(i) for i in range(1, 10))
    lesson_words = words + counting[:6]
    two = tuple(dict.fromkeys(' '.join(sentence.split()[:2]) + '.' for sentence in sentences))
    sources = dict(words=lesson_words, first_ten=lesson_words[:10], two=two,
                   sentences=sentences, four=four_word_sentences)
    rows = []
    for row in measured_rows():
        cases = [Question(text, row.get('answer', text))
                 for text in sources[row['corpus']]] if 'corpus' in row else []
        for case in row.get('cases', ()):
            if isinstance(case, str):
                cases.append(Question(case, case))
            else:
                cases.append(Question(case['text'], case['answer'], case.get('prompt'),
                                      tuple(case.get('context', ()))))
        rows.append((row['age'], row['measure'], tuple(cases)))
    return tuple(rows)
