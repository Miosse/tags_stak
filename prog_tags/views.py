from flask import Flask, render_template, request

app = Flask(__name__)

# On ajoute les configurations du serveur 
app.config.from_object('config')

from .utils import get_prediction2, recupere_post_id

# =============================================================================
# Les route à ajouter : 
#     - affichage de la page principale
#         - affiche liste-POSTS
#         - affiche le cadre de rédactions (Title + Body)
#         - affiche le cadre des tags générés
#         - pour la liste existante : afficher la liste des tags proposés
#             sur Stak Overflow
# 
# =============================================================================


   #########################
   #####   POINT D'ENTREE DU FICHIER ACTUEL
   #########################
@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
@app.route('/test5/', methods=['POST', 'GET'])
def index():
    dic_post = recupere_post_id(52799724)
    
    tags_text = '<fieldset><p><ul>'
    for i in dic_post['Tags']:
        #text += '<li><em>{}</em></li>'.format(i)
        tags_text += '<li>{}</li>'.format(i)
    
    tags_text += '</ul></p></fieldset>'
    
    return render_template('index2.html',
                           titre=dic_post['Title'],
                           msg=dic_post['Body'],
                           tags_values=tags_text,
                           tags2=dic_post['Tags'])



@app.route('/resultat_prediction', methods=['GET', 'POST'])
def resultat_prediction():
    my_prediction = get_prediction2(
        val1=request.form['post_title'], 
        val2=request.form['post_message'])
    
    text = '<fieldset><p><ul>'
    for i in my_prediction:
        text += '<li>{}</li>'.format(i)
    
    text += '</ul></p></fieldset>'
    return text


@app.route('/post_stackoverflow/<post_id>', methods=['GET', 'POST'])
def post_stackoverflow(post_id):
    
    post_id = post_id
    dic_post = recupere_post_id(int(post_id))
    
    tags_text = '<fieldset><p><ul>'
    for i in dic_post['Tags']:
        #text += '<li><em>{}</em></li>'.format(i)
        tags_text += '<li>{}</li>'.format(i)
    
    tags_text += '</ul></p></fieldset>'
    
    return render_template('post_existant.html', 
                           titre=dic_post['Title'],
                           msg=dic_post['Body'],
                           tags_values=tags_text,
                           tags2=dic_post['Tags']
                           )



@app.route('/test3', methods=['GET', 'POST'])
def test3():
    return render_template('redaction_post.html')


@app.route('/test4', methods=['GET', 'POST'])
def test4():
    return render_template('post_existant.html')

@app.route('/test6', methods=['GET', 'POST'])
def test6():
    dic_post = recupere_post_id(52799724)
    
    tags_text = '<fieldset><p><ul>'
    for i in dic_post['Tags']:
        #text += '<li><em>{}</em></li>'.format(i)
        tags_text += '<li>{}</li>'.format(i)
    
    tags_text += '</ul></p></fieldset>'
    
    return render_template('index2.html',
                           titre=dic_post['Title'],
                           msg=dic_post['Body'],
                           tags_values=tags_text,
                           tags2=dic_post['Tags'])




if __name__ == "__main__":
	app.run()

