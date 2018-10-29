/*
function completeAndRedirect(){
    alert('coool');
    
};

function check2() {
        var inputs = document.getElementsByTagName('input'),
            inputsLength = inputs.length;

        for (var i = 0; i < inputsLength; i++) {
            if (inputs[i].type === 'radio' && inputs[i].checked) {
                alert('La case cochée est la n°' + inputs[i].value);
            }
        }
    }

function stt1() {
	//console.log( "JE SUIS ICI function " );
     resultat = $.ajax({
     		type: "POST",
            url : 'http://127.0.0.1:5000/resultat_prediction', // on appelle le script JSON
            dataType : 'json', // on spécifie bien que le type de données est en JSON
            data : {say: "value1", user_message: "value2"

                
            },
            success : function(json) {
            		alert(cool);
                   //MA_MIN_DATE = json.date_min;
                   //MA_MAX_DATE = json.date_max;
                   //enableddates = json.dates_actives;
            },
            });


}

function stt2() {
	$.post(
		"{{ url_for('resultat_prediction') }}", 
		{ say: "value1", user_message: "value2" },
  		function(data) {
    		alert("Response: " + data);
  }
);
}

function stt4() {
	$.post(
		"{{ url_for('resultat_prediction') }}", 
		{ say: "value1", user_message: "value2" },
  		function(data) {
    		alert("Response: " + data);
  }
);
}
*/

function stt_new(m_url) {
    toggle('tags_Predits');
	$.post(
		m_url, 
		{ say: "value1", user_message: "value2" },
  		function(data) {
        		alert("Response_NEW: " + data);
  }
);
    toggle('tags_Predits');
}

function stt_new2(m_url) {
    toggle('tags_Predits');
	$.post(
		m_url, 
		{ say: "value1", user_message: "value2" },
  		function(data) {
        		alert("Response_NEW2: " + data);
        		$( ".inner" ).append( "<p>Test333333333</p>" );
        		$(".tags_Predits3").append( "<p>Test22222</p>" );
        		alert("AUTRE " + document.getElementById("tags_Predits3"))
        		toggle('tags_Predits');
        		
  }
);
}


function toggle(anId)
{
	node = document.getElementById(anId);
	if (node.style.visibility=="hidden")
	{
		// Contenu caché, le montrer
		//node.style.visibility = "visible";
		node.style.height = "auto";			// Optionnel rétablir la hauteur
	}
	else
	{
		// Contenu visible, le cacher
		node.style.visibility = "hidden";
		node.style.height = "0";			// Optionnel libérer l'espace
	}
}

$(document).ready(function(){
              $("#btn1").click(function(){
                  alert('TEST1 : ' + val1 + '//');
                  $("#test11").text(function(i, origText){
                      
                      return "Old text: " + val1 + origText + " New text: Hello world! (index: " + i + ")"; 
                  });
              });

              $("#btn2").click(function(){
                  $("#test2").html(function(i, origText){
                      return "Old html: " + origText + " New html: Hello <b>world!</b> (index: " + i + ")"; 
                  });
              });
          });
          
          
          
          
$(document).ready(function(){
              $("#btn1").click(function(){
                  
                  
                  $.post(
                		"http://127.0.0.1:5000/resultat_prediction", 
                		{ post_title: "post_title2", post_message: "post_message2" },
                		
                		
                  		function(data) {
                    		alert("Response: " + data);
                          }
                    );

                  
                  $("#tags_Predits1").html(function(i, origText){
                     return '<p><ul><li><em>TAG</em></li><li>TAG2</li></ul></p>';
                  });
                  
                  $("#tags_Predits3").text(function(i, origText){
                     return "Old text: " + origText + " New text: Hello world! (index: " + i + ")"; 
                  });
              });

              
          });
          
           
$(document).ready(function(){
              $("#btn1").click(function(){
                  
                  
                  $.post(
                		"http://127.0.0.1:5000/resultat_prediction", 
                		{ 
                    		post_title: $("#post_title").val(), 
                    		post_message: $("#post_message").val() 
                    	},
                		
                		
                  		function(data) {
                    		//alert("Response: " + data);
                    		$("#tags_Predits1").html(data);
                          }
                    );

              });

              
          });
          
           


$(document).ready(function(){
              $("#btn1").click(function(){
                  
                  
                  $.post(
                		"http://127.0.0.1:5000/resultat_prediction", 
                		{ 
                    		post_title: $("#post_title").val(), 
                    		post_message: $("#post_message").val() 
                    	},
                		
                		
                  		function(data) {
                    		alert("Response: " + data);
                    		$("#tags_Predits1").html(data);
                          }
                    );

                  
                  $("#tags_Predits1").html(function(i, origText){
                     return '<p><ul><li><em>TAG</em></li><li>TAG2</li></ul></p>';
                  });
                  
                  $("#tags_Predits3").text(function(i, origText){
                     return "Old text: " + origText + " New text: Hello world! (index: " + i + ")"; 
                  });
              });

              
          });



$(document).ready(function(){
              $("#btn1").click(function(){
                  alert('TESTICI');
                  $.post(
                		url_prediction, 
                		{ 
                    		post_title: $("#post_title").val(), 
                    		post_message: $("#post_message").val() 
                    	},
                		
                		function(data) {
                    		$("#tags_Predits1").html(data);
                          }
                    );
              });              
          });     
          
          