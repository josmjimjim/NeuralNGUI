from pylatex import (Document, Section, Command, Tabular, Figure, Center, Table)
from pylatex.utils import NoEscape

class Report:

    def __init__(self, param, model, img_path1, img_path2, pdf, log):
        self.param = param
        self.model = model
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.pdf = pdf
        self.log = log

    def generate_report(self):

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
                        tabular.add_row((r'Número de batches', self.param['Number of images in dataset']))
                        tabular.add_row((r'Clases    empleadas', self.param['Number of classes']))
                        tabular.add_row((r'Desviación estandar', self.param['Std']))
                        tabular.add_row((r'Media del dataset', self.param['Mean']))
                        tabular.add_row((r'Número de imágenes por clases', self.param['Number of images per class']))
                        tabular.add_row((r'Identificación de las clases', self.param['Class idx']))
                        tabular.add_empty_row()
                        tabular.add_hline()

        with doc.create(Section('Información sobre la red')):

            with doc.create(Figure(position='H')) as net:
                net.add_image(self.model, width='480px')
                net.add_caption('Estructura de la red')

        with doc.create(Section('Resultados del entrenamiento')):

            with doc.create(Figure(position='H')) as loss:
                loss.add_image(self.img_path1, width='480px')
                loss.add_caption('Resultados del entrenamiento')

            with doc.create(Figure(position='H')) as confusion:
                confusion.add_image(self.img_path2, width='480px')
                confusion.add_caption('Confusion matrix')

        with doc.create(Section('Log e información del entrenamiento')):
            with open(self.log, 'r') as file:
                logs = file.read()
                file.close()
            doc.append(logs)

        try:
            doc.generate_pdf(self.pdf, clean_tex=False, compiler='pdflatex')
        except Exception as e:
            print(e)



if __name__ == '__main__':

    try:
        pass

    except Exception as e:
        print(e)