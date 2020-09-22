# pitch_predictor
## Deep Learning Pitch Prediction Model 

![](southparkpitch.gif)

Did some work while studying in my Deep Learning class. We were learning about Artificial Neural Networks,
and I was looking up some practical applications that I could easily implement. 

I came across a project that attempted to predict pitches from statcast data based on various elements of a baseball game. 
These elements/features/variables include how many runners there are on base, the last pitch thrown, the handedness of the
batter and the pitcher etc, and attempted to predict the pitch type for a single pitcher. The writer bucketed the pitches
into general pitch types: Breaking, Fast, Changeup. 

I thought this particular approach was intriguing, and solved some of the problems around identifying pitches. Often, there
can be misclassifications between types of breaking ball pitches for example. 

I had two takeaways from reading this writer's analysis. 

  1. There was only one pitcher in the study. Why not more than that?
  
  2. I wish the writer had reported his confusion matrix, or accuracy for each class of pitch,
     because I suspect his accuracy was due to the model predicting fastball
     (the preferred pitch of the pitcher in question) for every pitch, thus getting a 60% accuracy. 

Anyway, here is my attempt at building a similar model, attempting to predict a breaking ball, fastball, or changeup during any at-bat in a 
Major League Baseball Game. 


