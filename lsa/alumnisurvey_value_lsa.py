#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Alumni Survey Project LSA KMEANS
#Cooper Project
#February 2017

import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import text 
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt, mpld3


filename = '/Users/jessicamarshall/Desktop/DataScienceIS/CUProject/datasets/value_cooper.xlsx'
data = pd.read_excel(filename, header = None, squeeze = 1)
#data_list = list(data.values.flatten())

print(data.isnull().values.any())

n_features = 20     #max 646
n_samples = data.size
n_components = 4     #for dimensionality reduction
k_means_clusters = 5   #default, can change

# Perform an IDF normalization on the output of HashingVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=1, stop_words = text.ENGLISH_STOP_WORDS.union(['cooper', 'students', 'faculty', 'free', 'value', 'critical', 'thinking', 'education', 'work', 'school', 'skills', 'experience', 'union', 'learning', 'think', 'tuition', 'professors', 'time', 'student', 'learn', 'small', 'community', 'ability', 'learned', 'problem', 'solving', 'life', 'art', 'ideas', 'body', 'institution', 'quality', 'engineering', 'environment', 'career', 'peers', 'strong', 'different', 'debt', 'creative', 'rigor', 'rigorous', 'diverse', 'working', 'classes', 'people', 'exposure', 'focus', 'good', 'helped', 'great', 'class', 'did', 'like', 'world', 'new', 'technical', 'prepared', 'scholarship', 'hard', 'years', 'taught', 'way', 'unique', 'critically', 'freedom', 'program', 'allowed', 'challenging', 'lot', 'able', 'having', 'academic', 'professional', 'valued', 'classmates', 'ethic', 'real', 'field', 'high', 'study', 'architecture', 'undergraduate', 'opportunity', 'valuable', 'problems', 'nyc', 'research', 'design', 'really', 'diversity', 'commitment', 'intelligent', 'intellectual' 'graduate', 'dedication', 'access', 'passionate', 'culture', 'appreciate', 'amazing', 'better', 'experiences', 'understanding', 'opportunities', 'artists', 'foundation', 'major', 'degree', 'course', 'difficult', 'smart', 'institutions', 'graduate', 'intellectual', 'merit', 'city', 'development', 'lab', 'schools', 'pursue', 'teaching', 'job', 'succeed', 'arts', 'values', 'explore', 'communication', 'attended', 'college', 'knowledge', 'practical', 'colleagues', 'teamwork', 'group', 'future', 'resources', 'information', 'provide', 'engaged', 'approach', 'fundamentals', 'practice', 'curriculum', 'educational', 'studies', 'artist', 'emphasis', 'tough', 'reputation', 'teachers', 'disciplines', 'engaging', 'talent', 'challenges', 'material', 'dedicated', 'excellent', 'support', 'unparalleled', 'challenged', 'truly', 'important', 'independent', 'best', 'interaction', 'didn', 've', 'talented', 'professor', 'leadership', 'teach', 'courses', 'projects', 'extremely', 'focused', 'helpful', 'independence', 'analytical', 'engagement', 'general', 'challenge', 'presentations', 'humanities', 'cu', 'perspective', 'computer', 'interdisciplinary', 'grad', 'especially', 'generally', 'humanities', 'incredible', 'brilliant', 'don', 'presentation', 'village', 'particularly', 'engineers', 'highly', 'importance', 'staff', 'civic', 'skill', 'demanding', 'artistic', 'atmosphere', 'graduating', 'fostered', 'fact', 'artistic', 'cost', 'writing', 'connections', 'critique', 'studio', 'discourse', 'instilled', 'thinker', 'curiosity', 'graduated', 'long', 'paid', 'coursework', 'background', 'provided', 'received', 'committed', 'higher', 'engineer', 'mentors', 'teacher', 'creativity', 'grateful', 'mission', 'breadth', 'status', 'collaboration', 'paying', 'shop', 'excellence', 'appreciation', 'programs', 'wealth', 'graduation', 'facilities', 'studios', 'undergrad', 'techniques', 'interviews', 'creatively', 'competitive', 'project', 'resume', 'invaluable']), use_idf= True)

X = vectorizer.fit_transform(data)

print("n_samples: %d, n_features: %d" % X.shape)
print()

dist = 1 - cosine_similarity(X)
print(dist)
print()

print("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.

####
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)
####

#explained_variance = svd.explained_variance_ratio_.sum()
#print("Explained variance of the SVD step: {}%".format(
#        int(explained_variance * 100)))

print()

#DO KMEANS
km = KMeans(n_clusters=k_means_clusters, init='k-means++', max_iter=100, n_init=10)
km.fit(X)

########## PRINT TERMS and CLUSTERS
from sklearn.externals import joblib
joblib.dump(km,  'doc_cluster.pkl')

clusters = km.labels_.tolist()

responses = {'text': data, 'cluster': clusters }
#dictionary

frame = pd.DataFrame(responses, index = None)
print(frame['cluster'].value_counts())

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid

original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(k_means_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :10]: #replace 10 with n words per cluster   #number of terms
        print(' %s' % terms[ind], end='')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for response in frame[frame.cluster == i].text.values.tolist():
        print(' %s,' % response, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()

####DIMENSIONALITY REDUCTION

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

##########VIS

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: '0', 
                 1: '1', 
                 2: '2', 
                 3: '3', 
                 4: '4'}


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=frame.cluster, title=frame.text))

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot 

fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.show() #show the plot

#html = mpld3.fig_to_html(fig)
#print(html)