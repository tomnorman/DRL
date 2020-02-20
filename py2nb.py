#!/usr/bin/env python3

##ipynb to py
#jupyter nbconvert --to python Index.ipynb

#py to ipynb
import os
import nbformat.v4
import argparse

starts = ('#Title#','#Markdown#','#Code#')

TITLE,MARKDOWN,CODE = [i for i in range(len(starts))]


def add_cell(nb, cell_str, cell_type):
    if cell_str:
        if cell_str[-2] == '\n': #remove trailing new lines
            cell_str = cell_str[:-2]
        if cell_type == TITLE:
            cell_str = cell_str.replace('#', '##')
            cell = nbformat.v4.new_markdown_cell(cell_str)
            cell.metadata.heading_collapsed = True
        elif cell_type == MARKDOWN:
            cell_str = cell_str.replace('# ', '')
            cell = nbformat.v4.new_markdown_cell(cell_str)
        else:
            cell = nbformat.v4.new_code_cell(cell_str)
        nb.cells.append(cell)


def convert(file_name):
    """ Convert the python script to jupyter notebook"""
    with open(file_name) as f:
        cell_str = ''
        cell_type = None
        nb = nbformat.v4.new_notebook()
        for line in f:
            if line.startswith(starts):
                #add cell_str
                add_cell(nb,cell_str,cell_type)
                for i,start in enumerate(starts):
                    if line.startswith(start):
                        cell_type = i
                        break
                #init new cell_str
                cell_str = ''
                continue

            if cell_type is not None:
                cell_str += line

        add_cell(nb,cell_str,cell_type) #add last cell
        notebook_name = os.path.splitext(file_name)[0] + '.ipynb'
        nbformat.write(nb, notebook_name)


def parse_args():
    description = \
"Convert python script to jupyter notebook.\n\
How to use:\n\
insert one of: {} to add a cell\n\n\
example:\n______\n\
#Code#\nprint('hello world')\n______\n\
will add code block with print command".format(','.join(starts))
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("file_name", help="convert \'file_name.py\' to \'file_name.ipynb\'")
    return parser.parse_args() 


def main():
    args = parse_args()
    if args.file_name.endswith('.py'):
        convert(args.file_name)
    else:
        print('supports only .py ending')


if __name__ == '__main__':
    main()
