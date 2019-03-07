#-*- coding: utf-8 -*-

import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


colors = ['#EF476F', '#FFD166', '#06D6A0', '#118AB2', '#073B4C']

def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice


def html_header():
    header = """
    <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"> 
    <html>

    <head>
      <meta name=viewport content=“width=650”>
      <meta name="generator" content="HTML Tidy for Linux/x86 (vers 11 February 2007), see www.w3.org">
      <style type="text/css">
        /* Color scheme stolen from Sergey Karayev */

        a {
          color: #1772d0;
          text-decoration: none;
        }

        a:focus,
        a:hover {
          color: #f09228;
          text-decoration: none;
        }

        body,
        td,
        th,
        tr,
        p,
        a {
          font-family: "Lato", Verdana, Helvetica, sans-serif;
          font-size: 14px
        }

        strong {
          font-family: "Lato", Verdana, Helvetica, sans-serif;
          font-size: 14px;
        }

        heading {
          font-family: "Lato", Verdana, Helvetica, sans-serif;
          font-size: 22px;
        }

        papertitle {
          font-family: "Lato", Verdana, Helvetica, sans-serif;
          font-size: 14px;
          font-weight: 700
        }

        name {
          font-family: "Lato", Verdana, Helvetica, sans-serif; 
          }

        .one {
          width: 160px;
          height: 160px;
          position: relative;
        }

        .two {
          width: 160px;
          height: 160px;
          position: absolute;
          transition: opacity .2s ease-in-out;
          -moz-transition: opacity .2s ease-in-out;
          -webkit-transition: opacity .2s ease-in-out;
        }

        .fade {
          transition: opacity .2s ease-in-out;
          -moz-transition: opacity .2s ease-in-out;
          -webkit-transition: opacity .2s ease-in-out;
        }

        .my {
          border: 1px solid;
          width: 650px; 
          align: "center"; 
          cellspacing: 0;
          cellpadding: 0; 
        }  



        span.highlight {
          background-color: #ffffd0;
        }
      </style>
      <link rel="icon" type="image/png" href="seal_icon.png">
      <title>CNN-Units-in-NLP</title>
      <meta http-equiv="Content-Type" content="text/html; charset=us-ascii">
      <link href="http://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel='stylesheet' type='text/css'>
    </head>
    <body>

    <table width="650" border="0" align="center" cellspacing="0" cellpadding="0">
        <tr>
            <td align="right">[#] denotes morpheme concept</td>
        </tr>
        <tr>
            <td align="right"> M=3 concepts are aligned per unit
        </tr>
    """
    return header


def html_per_unit(task, layer, unit, alignment, num_align):
    html="""
    <tr>
        <td align="left">[%s / layer %02d / Unit %04d]<br>
    """ % (task, layer, unit)

    for i in range(num_align):
        concept, doa = alignment[unit][i]
        concept = concept.replace('MORPH_', '[#]')

        html += '<span style="background-color: %s" >%s</span> (%.2lf) / ' % (colors[i], concept, doa)
    html += "</tr>"

    return html.decode("utf-8", errors="ignore")
    
def html_per_tas(unit, tas, alignment, num_visualized_tas, max_sent_len):
    html="""
    <tr>
        <td>
            <table class=my>
    """

    for activ, sentence in tas[unit][:num_visualized_tas]:
        if len(sentence) > max_sent_len:
            sentence = sentence[:max_sent_len] + ' (...)'

       
        
        for idx, (concept, _) in enumerate(alignment[unit]):
            concept = concept.replace('MORPH_', '')

            # set max length for visualization
            concept = concept[:25]

            try:
                if concept.lower() in sentence.lower():
                    s_idx = sentence.lower().index(concept.lower())
                    sentence = \
                            sentence[:s_idx] + \
                            '<span style="background-color: %s" >' % colors[idx] + \
                            sentence[s_idx: s_idx + len(concept)] + \
                            '</span>' + \
                            sentence[s_idx + len(concept):]

            except:
                #from IPython import embed; embed()
                continue                 

        html += """
        <tr>
            <td align="left"><li>{}</li></td>
        </tr>
        """.format(sentence)

    html += """
            </table>
        </td>
    </tr>

    <tr>
        <td><br></td>
    </tr>
    """
    return html.decode('utf-8', errors='ignore')



