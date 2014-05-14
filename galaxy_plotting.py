import numpy as np
import matplotlib

from tsne import bh_sne
from mpld3 import plugins, utils
from matplotlib import pyplot as plt

def plot_projection(X, y, tsne_trigger=True):
    """Plots a reduced version of the data with t-sne.

    Attributes:
        X: The feature vectors.
        y: The labels.
    """
    galaxy_ids = y.index.values.tolist()
    classinfo = y.values.tolist()

    if tsne_trigger:
        X_projected = bh_sne(X.astype(np.float64))
    else:
        X_projected = X

    (fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    points = ax[0].scatter(X_projected[:,0], X_projected[:,1], s=300, alpha=0.3, c=y, cmap=plt.cm.autumn)
    
    for axis in [ax[1].xaxis, ax[1].yaxis]:
        axis.set_ticks([])
        axis.set_major_formatter(plt.NullFormatter())
    
    loadgalaxy = Load_Galaxy(points, galaxy_ids, classinfo)
    plugins.connect(fig, loadgalaxy)

class Load_Galaxy(plugins.PluginBase):
    JAVASCRIPT = """
    mpld3.register_plugin("loadgalaxy", LoadGalaxyPlugin);
    LoadGalaxyPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    LoadGalaxyPlugin.prototype.constructor = LoadGalaxyPlugin;
    LoadGalaxyPlugin.prototype.requiredProps = ["idpts", "galaxyids", "classinfo"];    
    function LoadGalaxyPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    LoadGalaxyPlugin.prototype.draw = function(){
        var obj = mpld3.get_element(this.props.idpts, this.fig);
        var galaxyids = this.props.galaxyids;
        var classinfo = this.props.classinfo
    
        var galaxyimage = obj.fig.canvas.append("image")
            .attr("width", 424)
            .attr("height", 424)
            .attr("x", obj.fig.axes[1].position[0])
            .attr("y", obj.fig.axes[1].position[1]);

        var galaxyclass1 = obj.fig.canvas.append("text")
            .style("font-size", 14)
            .style("text-anchor", "right")
            .attr("x", obj.fig.axes[1].position[0] + 50)
            .attr("y", obj.fig.axes[1].position[1] + obj.fig.axes[1].height + 20)
            
        var galaxyclass2 = obj.fig.canvas.append("text")
            .style("font-size", 14)
            .style("text-anchor", "right")
            .attr("x", obj.fig.axes[1].position[0] + 50)
            .attr("y", obj.fig.axes[1].position[1] + obj.fig.axes[1].height + 40)
            
        var galaxyclass3 = obj.fig.canvas.append("text")
            .style("font-size", 14)
            .style("text-anchor", "right")
            .attr("x", obj.fig.axes[1].position[0] + 50)
            .attr("y", obj.fig.axes[1].position[1] + obj.fig.axes[1].height + 60)
    
        var tooltip = d3.select("body")
            .append("div")
            .style("position", "absolute")
            .style("z-index", "10")
            .style("width","60px")                  
            .style("height","28px")                 
            .style("padding","2px")             
            .style("font","12px sans-serif")
            .style("border","0px")      
            .style("border-radius","8px")  
            .style("background", "lightsteelblue")
            .style("visibility", "hidden");
    
        function mouseover(d, i){
            var galaxyid = galaxyids[i]
            tooltip.text("GalaxyID: " + galaxyid);
            tooltip.append("img")
            .attr("src", "input_data/images_training_rev1/" + galaxyid + ".jpg")
            .attr("x", -8)
            .attr("y", -8)
            .attr("width", 500)                  
            .attr("height", 500); 
            tooltip.style("visibility", "visible");
        }
        
        function mousemove(){
            return tooltip.style("top", (event.pageY-10)+"px").style("left",(event.pageX+10)+"px");
        }
        
        function mouseout(){
            return tooltip.style("visibility", "hidden");
        }
        
        function mousedown(d, i){
            var galaxyid = galaxyids[i]
            var y = classinfo[i]

            galaxyimage.attr("xlink:href", "input_data/images_training_rev1/" + galaxyid + ".jpg")
            galaxyclass1.text("Class1.1:     " + y[0])
            galaxyclass2.text("Class1.2:     " + y[1])
            galaxyclass3.text("Class1.3:     " + y[2])
        }
    
        obj.elements()
            .on("mousedown", mousedown)
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseout", mouseout);
        };
    """

    def __init__(self, points, galaxyids, classinfo):
        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None

        self.dict_ = {"type": "loadgalaxy",
                      "idpts": utils.get_id(points, suffix),
                      "galaxyids": galaxyids,
                      "classinfo": classinfo}
