#!/usr/bin/perl

# commands for latex,bibtex,dvipdf. %S : source file, %B : base name, %D : pdf file, %O : option
$latex = 'platex %O -interaction=nonstopmode -synctex=1 -kanji=utf8 %S';
$bibtex = 'pbibtex %O %B -kanji=utf8';
$dvipdf = 'dvipdfmx %O -o %D %S';

# method to make pdf - 0:not making, 1:pdflatex, 2:ps2pdf, 3:dvipdf
$pdf_mode = 3;
