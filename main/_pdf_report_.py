from pylatex import Document, Section, Command, Tabular, Figure
from pylatex.utils import NoEscape


def generate_report():

    doc = Document()
    # Define tittle and preamble
    doc.preamble.append(Command('title', 'Informe de entrenamiento de la red'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Información sobre el dataset')):

        doc.append('En la siguiente tabla se recoge toda la información \
         relacionada, sobre el dataset empleado en el entrenamiento de la red neuronal: ')

        with doc.create(Tabular('c|c')) as table:
            table.add_hline()
            table.add_row((1, 2))
            table.add_empty_row()
            table.add_row((4, 5))
            table.add_hline()

    with doc.create(Section('Información sobre la red')):

        doc.append('Lorem ipsum')

    with doc.create(Section('Resultados del entrenamiento')):

        with doc.create(Figure(position='h!')) as loss:
            loss.add_image('image_filename', width='120px')
            loss.add_caption('Look it\'s on its back')

        with doc.create(Figure(position='h!')) as confusion:
            confusion.add_image('image_filename', width='120px')
            confusion.add_caption('Look it\'s on its back')

    with doc.create(Section('Logs y salidas del proceso')):
        doc.append('Lorem ipsum 2')

    return doc


if __name__ == '__main__':
   doc = generate_report()
   doc.generate_pdf('basic_maketitle', clean_tex=False, compiler='pdflatex')