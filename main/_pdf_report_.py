import os
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape


def generate_report():

    doc = Document()
    # Define tittle and preamble
    doc.preamble.append(Command('title', 'Neural Network Training Report'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Dataset Information')):

        doc.append('''Lorem ipsum
        Lorem ipsum 2
        ''')
    return doc


if __name__ == '__main__':
   doc = generate_report()
   doc.generate_pdf('basic_maketitle', clean_tex=False)