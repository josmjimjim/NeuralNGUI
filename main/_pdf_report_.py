from pylatex import (Document, Section, Command, Tabular, Figure, Center, Table)
from pylatex.utils import NoEscape
import argparse

def generate_report():

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
                    tabular.add_row((r'Número de imágenes', 2))
                    tabular.add_row((r'Clases empleadas', 5))
                    tabular.add_row((r'Desviación estandar', 5))
                    tabular.add_row((r'Media del dataset', 5))
                    tabular.add_row((r'Número de imágenes por clase', 5))
                    tabular.add_row((r'Identificación de las clases', 5))
                    tabular.add_empty_row()
                    tabular.add_hline()

    with doc.create(Section('Información sobre la red')):

        doc.append('')

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
   try:
        doc.generate_pdf('basic_maketitle', clean_tex=False, compiler='pdflatex')
   except Exception as e:
       print(e)


    # Define args to pass to the model
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Select the model for training')
    parser.add_argument('optimizer', help='Select the optimizer')
    parser.add_argument('batch', help='Define the batch size')
    parser.add_argument('epochs', help='Define the epochs for training model')
    parser.add_argument('lr', help='Define learning rate for the optimizer')
    parser.add_argument('save', help='Select file directory to save model')
    parser.add_argument('train', help='Select file directory of training images')
    parser.add_argument('-t', '--test', help='Select file directory of test images')
    parser.add_argument('-w', '--weights', help='(Optional) Select pretrained weights')

    args = parser.parse_args()

    if args.weights:
        weights = args.weights
    else:
        weights = None

    if args.test:
        test = args.test
    else:
        test = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neural = NeuralNetwork(args.model, args.optimizer, int(args.batch),
                      int(args.epochs), float(args.lr), args.save,
                      args.train, test, weights
                     )

    neural.train_model(device)
