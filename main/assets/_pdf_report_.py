from pylatex import (Document, Section, Command, Tabular, Figure, Center, Table)
from pylatex.utils import NoEscape

class Report:

    def __init__(self, param, model, img_path1, img_path2, img_path3, pdf, log):
        self.param = param
        self.model = model
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.img_path3 = img_path3
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
                        tabular.add_empty_row()
                        tabular.add_hline()

        with doc.create(Section('Resultados del entrenamiento')):

            with doc.create(Figure(position='h!')) as loss:
                loss.add_image(self.img_path1, width=NoEscape(r'0.45\linewidth'))
                loss.add_caption('Resultados del entrenamiento. Pérdida')

            with doc.create(Figure(position='h!')) as acc:
                acc.add_image(self.img_path2, width=NoEscape(r'0.45\linewidth'))
                acc.add_caption('Resultados del entrenamiento. Precision')

            with doc.create(Figure(position='h!')) as confusion:
                confusion.add_image(self.img_path3, width=NoEscape(r'0.45\linewidth'))
                confusion.add_caption('Confusion matrix')

        Command('newpage')

        with doc.create(Section('Información sobre la red')):
            with doc.create(Figure(position='h!')) as net:
                Command('includegraphics', options=['width=\linewidth', 'height=20cm',
                'keepaspectratio'], arguments=self.model).dumps()
                net.add_caption('Estructura de la red')

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