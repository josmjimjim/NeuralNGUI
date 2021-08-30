from pylatex import (Document, Section, Command, Tabular, Figure, Center, Table)
from pylatex.utils import NoEscape
import argparse

def generate_report(param, model, img_path1, img_path2):

    geometry_options = {
        "top": "0.8in",
        "margin": "0.8in",
        "bottom": "0.8in",
    }

    doc = Document(geometry_options=geometry_options)

    # Define tittle and preamble
    doc.preamble.append(Command('title', 'Informe de entrenamiento de la red'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Información sobre el dataset')):

        doc.append('En la siguiente tabla se recoge toda la información \
        relacionada con el dataset empleado durante el entrenamiento de la red neuronal: \n')

        with doc.create(Table(position='htp')) as table:
            table.add_caption('Resumen propiedades del dataset')
            with table.create(Center()) as centered:
                with centered.create(Tabular('c|c')) as tabular:
                    tabular.add_hline()
                    tabular.add_row((r'Número de imágenes', param['Number of images in dataset']))
                    tabular.add_row((r'Clases empleadas', param['Number of classes']))
                    tabular.add_row((r'Desviación estandar', param['Std']))
                    tabular.add_row((r'Media del dataset', param['Mean']))
                    tabular.add_row((r'Número de imágenes por clases', param['Number of images per class']))
                    tabular.add_row((r'Identificación de las clases', param['Class idx']))
                    tabular.add_empty_row()
                    tabular.add_hline()

    with doc.create(Section('Información sobre la red')):

        doc.append(model)

    with doc.create(Section('Resultados del entrenamiento')):

        with doc.create(Figure(position='h!')) as loss:
            loss.add_image(img_path1, width='120px')
            loss.add_caption('Look it\'s on its back')

        with doc.create(Figure(position='h!')) as confusion:
            confusion.add_image(img_path2, width='120px')
            confusion.add_caption('Look it\'s on its back')

    return doc


if __name__ == '__main__':
    # Define args to pass to the model
    parser = argparse.ArgumentParser()
    parser.add_argument('param', help='Dictionary of parameters')
    parser.add_argument('model', help='Model Structure')
    parser.add_argument('loss', help='Loss path')
    parser.add_argument('cfm', help='Confusion matrix path')
    parser.add_argument('pdf', help='Report saving path')
    args = parser.parse_args()

    try:
        doc = generate_report(args.param, args.model, args.loss, args.cfm)
        doc.generate_pdf(args.pdf, clean_tex=False, compiler='pdflatex')
    except Exception as e:
        print(e)