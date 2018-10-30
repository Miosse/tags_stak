from flask import Flask, render_template, request

app = Flask(__name__)

# On ajoute les configurations du serveur 
app.config.from_object('config')

from .utils import get_prediction2, recupere_post_id

   #########################
   #####   POINT D'ENTREE DU FICHIER ACTUEL
   #########################
@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    dic_post = recupere_post_id(52799724)
    
    tags_text = '<fieldset><p><ul>'
    for i in dic_post['Tags']:
        tags_text += '<li>{}</li>'.format(i)
    
    tags_text += '</ul></p></fieldset>'
    
    return render_template('index2.html',
                           titre=dic_post['Title'],
                           msg=dic_post['Body'],
                           tags_values=tags_text,
                           tags2=dic_post['Tags'])

   #########################
   #####   API 
   #########################

@app.route('/resultat_prediction', methods=['GET', 'POST'])
def resultat_prediction():
    '''API de prédiction d'un POST : récupère les informations de 
    titre et de message via les variables POST post_title et post_message
    INPUT:
    ------
        - Variable POST : post_title avec le texte du titre
        - Variable POST : post_message avec le texte du message
    OUTPUT:
    ------
        - Renvoie un text avec les balises HTML 
    '''
    my_prediction = get_prediction2(
        val1=request.form['post_title'], 
        val2=request.form['post_message'])
    
    text = '<fieldset><p><ul>'
    for i in my_prediction:
        text += '<li>{}</li>'.format(i)
    
    text += '</ul></p></fieldset>'
    return text


   #########################
   #####   Affichage d'un POST existant
   #########################

@app.route('/post_stackoverflow/<post_id>', methods=['GET', 'POST'])
def post_stackoverflow(post_id):
    '''Affiche un post existant : 
        si le post_id est invalide alors un numéro aléatoire est généré'''
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

   #########################
   #####   Affichage d'un POST aléatoire
   #########################

@app.route('/random', methods=['GET', 'POST'])
@app.route('/post', methods=['GET', 'POST'])
def random_post_stackoverflow():
    '''Affiche aléatoirement un post existant'''
    dic_post = recupere_post_id()
    
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

   #########################
   #####   Affichage d'une interface de rédaction de post
   #########################

@app.route('/create', methods=['GET', 'POST'])
def create_post():
    return render_template('redaction_post.html')



if __name__ == "__main__":
	app.run()

