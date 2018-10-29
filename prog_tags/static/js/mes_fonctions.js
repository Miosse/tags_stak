      $(document).ready(function(){
              $("#tags_stacksOverflow").html(tags_values);
              
              $("#btn2").click(function(){
                  $.post(
                		url_prediction, 
                		{ 
                    		post_title: $("#post_title_existant").val(), 
                    		post_message: $("#post_message_existant").val() 
                    	},
                		
                		function(data) {
                    		$("#tags_Predits2").html(data);
                    		
                          }
                    );
              }); 
              
               $("#btn1").click(function(){
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
