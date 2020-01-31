d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

var getMax = function (data) {
    var max = 0;
    Graph.sources.forEach(source => {
        data[source].forEach(record => {
            var rent = record.median_rent;
            var moe = record.median_rent_moe;
            if(moe == 'null'){
                moe = 0
            }
            if (moe + rent > max){
                max = moe + rent
            }
        })
    })
    return max;
}

var Graph = {
    sources: ['zillow','acs5','acs1','safmr'],
    source_name_full: {
        'safmr':'Fair market rent',
        'acs1':'1-year ACS',
        'acs5':'5-year ACS',
        'zillow':'Zillow'
    },
    color_dict: {
        'safmr':'#fdbf11',
        'acs1':'#0a4c6a',
        'acs5':'#5c5859',
        'zillow':'#1696d2'
    },
    margin: {top: 40, right: 30, bottom: 20, left: 50},
    h: 450,
    y_max: 3000,
    year: 2015,
    newZIPCode: function() {
        var newZip = document.getElementById("ZIPCodeInput").value;
        if (newZip) {
            var newData = this.dataMap[newZip];
            this.updateGraph(newData);
        };
    },

    initGraph: function(data) {
        this.sources.forEach((source) => {
            if (['zillow','safmr'].includes(source)) {
                this.svg.append("path")
                    .datum(data[source]) // 10. Binds data to the line
                    .attr("class", source + " line") // Assign a class for styling
                    .attr('stroke-dasharray',"4 3")
                    .attr("d", this.valueline) // 11. Calls the line generator
            } else {
                this.svg.append("path")
                    .datum(data[source])
                    .attr('class', source + " error")
                    .attr("d", this.error)
                this.svg.append("path")
                    .datum(data[source]) // 10. Binds data to the line
                    .attr("class", source + " line no-error") // Assign a class for styling
                    .attr("d", this.valueline) // 11. Calls the line generator
            }
        });
        d3.select('.legend').moveToFront();

        // add vertical line
        var mouseG = this.svg.append("g")
            .attr("class", "mouse-over-effects");

        mouseG.append("path") // this is the black vertical line to follow mouse
            .attr("class", "mouse-line")
        var circle_data = Object.keys(data).map(function(key) {
            return [key, data[key]];
        });
        var mousePerLine = mouseG.selectAll('.mouse-per-line')
            .data(circle_data)
            .enter()
            .append("g")
            .attr("class", "mouse-per-line")
            .style("opacity", 0);

        mousePerLine.append("circle")
            .attr("r", 5)

        // add text box
        year_data = [this.year]

        this.legend.selectAll('.text-square')
            .data(circle_data)
            .enter()
            .append("rect")
                .attr('x', 20)
                .attr('height', 10)
                .attr('width', 10)
                .attr("class", "text-square")
        text_estimate = this.legend.selectAll('.text-estimate')
            .data(circle_data)
            .enter()
            .append("text")
                .attr('x', 40)
                .attr('font-size', 12)
                .attr("class", "text-estimate")
        text_estimate.append('tspan')
            .attr('class','text-source')
            .attr('font-weight','bold')
        text_estimate.append('tspan')
            .attr('class','text-rent')
    },

    updateGraph: function(data) {
        var svg = d3.select("body").transition()
        var max = getMax(data);
        if (max > this.y_max) {
            this.y.domain([0, max]);
        } else {
            this.y.domain([0, this.y_max]);
        };
        svg.select(".y")
            .transition(0)
                .call(this.customYAxis);
        this.sources.forEach(source => {
            if (!['zillow','safmr'].includes(source)){
                svg.select("."+source+".error")
                    .duration(750)
                    .attr("d", this.error(data[source]))
            }
            svg.select("."+source+".line")
                .duration(750)
                .attr("d", this.valueline(data[source]))
        });

        var circle_data = Object.keys(data).map(function(key) {
            return [key, data[key]];
        });

        var mousePerLine = this.svg.selectAll('.mouse-per-line')
            .data(circle_data)
            .enter()
            .append("g")
            .attr("class", "mouse-per-line")
            .style("opacity", 0);

        mousePerLine.exit().remove();
        this.drawDots(applyTransition = true);

        this.legend.selectAll('.text-square')
            .data(circle_data)
            .enter()
            .append("rect")
                .attr('x', 20)
                .attr('height', 10)
                .attr('width', 10)
                .attr("class", "text-square")
            .exit().remove()
        text_estimate = this.legend.selectAll('.text-estimate')
            .data(circle_data)
            .enter()
            .append("text")
                .attr('x',40)
                .attr('font-size', '12')
                .attr("class", "text-estimate")
        text_estimate.append('tspan')
            .attr('class','text-source')
            .attr('font-weight','bold')
        text_estimate.append('tspan')
            .attr('class','text-rent')
        text_estimate.exit().remove();
        this.insertText();
    },

    insertText: function() {
        var rents_for_year_by_source = d3.selectAll(".text-estimate").data()
            .map((series) => {
                var record = series[1].find((element)=> {
                    return element.time == Graph.year
                })
                if (record) {
                    return [series[0], record.median_rent]
                } else {
                    return [series[0], 0]
                }
            })
        rents_for_year_by_source = rents_for_year_by_source.sort(function(a,b){
            return b[1] - a[1];
        })

        var y_pos_arr = [0,1,2,3].map(int => {return 25+int*20})
        var y_pos_dict = {};
        rents_for_year_by_source.forEach((key, i) => y_pos_dict[key[0]] = y_pos_arr[i]);
        d3.selectAll(".text-estimate")
            .transition()
                .duration(500)
            .attr('y', function(d,i){
                return y_pos_dict[d[0]]
            })
        d3.selectAll(".text-source")
            .text(function(d, i) {
                return Graph.source_name_full[d[0]]
            })
        d3.selectAll(".text-rent")
            .text(function(d, i) {
                var rent = rents_for_year_by_source.find((e) => {return e[0] == d[0]})[1];
                if (rent > 0) {
                    return ": "+d3.format("$,")(rent)+"/month";
                } else {
                    return ": No data";

                }
            })
            .attr('fill',function(d, i) {
                var rent = rents_for_year_by_source.find((e) => {return e[0] == d[0]})[1];
                if (rent > 0) {
                    return "#000000";
                } else {
                    return "#d2d2d2";
                }
            })
        d3.selectAll(".text-square")
            .transition()
                .duration(500)
            .attr('fill', function(d, i) {return Graph.color_dict[d[0]]})
            .attr('y', function(d, i) {return y_pos_dict[d[0]]-9})
    },

    drawDots: function(applyTransition = false) {
        if (applyTransition) {
            var transitionTime = 750
        } else {
            var transitionTime = 0
        }
        d3.selectAll(".mouse-per-line")
            .transition()
                .duration(transitionTime)
            .attr("transform", function(d, i) {
                var record_for_year = d[1].find(function(element) {
                    return element.time == Graph.year;
                });
                if (record_for_year) {
                    return "translate(" + Graph.x(Graph.year) + "," + Graph.y(record_for_year.median_rent) +")";
                }
            })
            .style("opacity", function(d,i) {
                var record_for_year = d[1].find(function(element) {
                    return element.time == Graph.year;
                });
                if (record_for_year) {
                    return 1;
                } else {
                    return 0
                }
            })
        d3.select('mouse-over-effects').moveToFront();
        Graph.insertText(applyTransition);
    },

    drawVerticalLine: function() {
        try {
            var mouse = d3.mouse(this);
            var x_val = mouse[0]
            if (Graph.year != Math.round(Graph.x.invert(x_val))){
                Graph.year = Math.round(Graph.x.invert(x_val))
                d3.select(".mouse-line")
                    .attr("d", function() {
                        var d = "M" + Graph.x(Graph.year) + "," + Graph.height;
                        d += " " + Graph.x(Graph.year) + "," + 0;
                        return d;
                    });
                d3.select('.legend').moveToFront()
                Graph.drawDots();
            }
        }
        catch (TypeError) { // non-standard
            d3.select(".mouse-line")
                .attr("d", function() {
                    var d = "M" + Graph.x(Graph.year) + "," + Graph.height;
                    d += " " + Graph.x(Graph.year) + "," + 0;
                    return d;
                });
            d3.select('.legend').moveToFront()
            Graph.drawDots();
        }
    },

    drawLegend: function() {
        var x_start = 20
        var length = 20
        var y = 105

        this.legend.append("line")
            .attr("class", "line") // Assign a class for styling
            .attr('stroke-dasharray',"4 3")
            .attr('x1',x_start)
            .attr('y1', y)
            .attr('x2',x_start + length)
            .attr('y2', y)
            .attr('stroke','#5c5859')
            .attr('stroke-dasharray',"4 3")

        this.legend.append("text")
            .attr("class", "legend-text")
            .attr("text-anchor", "start")
            .attr("alignment-baseline", "middle")
            .attr("font-size", 12)
            .attr("x", x_start + length + 10)
            .attr("y", y+1)
            .text("No margin of error");

        this.legend.append("rect")
            .attr("class", "error") // Assign a class for styling
            .attr('x',x_start)
            .attr('y', y + 20 - length/2)
            .attr('width',length)
            .attr('height', length)
            .attr('fill','#5c5859')
            .attr('fill-opacity',0.5)

        this.legend.append("line")
            .attr("class", "line") // Assign a class for styling
            .attr('x1',x_start)
            .attr('y1', y + 20)
            .attr('x2',x_start + length)
            .attr('y2', y+20)
            .attr('stroke','#5c5859')

        this.legend.append("text")
            .attr("class", "legend-text")
            .attr("text-anchor", "start")
            .attr("alignment-baseline", "middle")
            .attr("font-size", 12)
            .attr("x", x_start + length + 10)
            .attr("y", y+21)
            .text("Margin of error");
    },

    drawGraph: function (containerWidth) {
        // console.log(containerWidth)
        $("#fig-container").empty();
        $("#custom-select").empty();
        $("#legend-container").empty();
        if (containerWidth > 558) {
            $("#legend-container").css('left', containerWidth - 250);
        } else {
            $("#legend-container").css('left', -20);
        }
        $(".title").css('width', containerWidth - 275);


        // var containerWidth = window.i
        var width = containerWidth - this.margin.left - this.margin.right,
            height = this.h - this.margin.top - this.margin.bottom;
        this.width = width
        this.height = height

        // set the ranges
        var x = d3.scaleLinear().range([0, width]);
        var y = d3.scaleLinear().range([height, 0]);

        this.x = x;
        this.y = y;

        x.domain([2010,2020]);
        y.domain([0,this.y_max]);

        // append the svg obgect to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        this.svg = d3.select("#fig-container").append("svg")
            .attr("width", width + this.margin.left + this.margin.right)
            .attr("height", height + this.margin.top + this.margin.bottom)
          .append("g")
            .attr("transform",
                  "translate(" + this.margin.left + "," + this.margin.top + ")");

        this.svg.append('rect')
            .attr("width", width + this.margin.left + this.margin.right)
            .attr("height", height + this.margin.top + this.margin.bottom)
            .attr('fill','white');

        this.legend = d3.select("#legend-container").append('svg')
            .style('width','400px')

        // Add the X Axis
        xAxis = d3.axisBottom(x)
            .tickFormat(d3.format(".4"));

        this.svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .attr("class", "x")
            .call(xAxis);

        // Add the Y Axis
        if(this.dataMap){
            var data = this.dataMap[document.getElementById("ZIPCodeInput").value];
            var svg = d3.select("body").transition();
            var max = getMax(data);
            if (max > this.y_max) {
                this.y.domain([0, max]);
            } else {
                this.y.domain([0, this.y_max]);
            };
        }
        yAxis = d3.axisRight(y)
            .tickSize(width)
            .tickFormat(d3.format("$,"));
        this.customYAxis= function (g) {
            g.call(yAxis);
            g.select(".domain").remove();
            g.selectAll(".tick:not(:first-of-type) line")
                .attr("stroke", "#DEDDDD")
                // .attr("stroke-dasharray", "2,2");
            g.selectAll(".tick text")
                .attr("x", -10)
                .attr("text-anchor", 'end');
        }
        this.svg.append("g")
            .attr("class", "y")
            .call(this.customYAxis)

        this.svg.append("text")
            .attr("class", "y-label")
            .attr("text-anchor", "start")
            .attr("y", -20)
            .attr("x", -this.margin.left)
            .text("Rent per month estimate");

        this.valueline = d3.line()
            .x(function(d) { return x(d.time); })
            .y(function(d) { return y(d.median_rent); });

        this.error = d3.area()
            .x(function(d) {return x(d.time); })
            .y0(function(d) { return y(d.median_rent-d.median_rent_moe); })
            .y1(function(d) { return y(d.median_rent+d.median_rent_moe); });

        this.svg.on("touchmove mousemove", this.drawVerticalLine)

        if(this.dataMap){
            var initialData = this.dataMap[document.getElementById("ZIPCodeInput").value];
            this.initGraph(initialData)
            this.drawLegend();
            this.insertText();
            this.drawVerticalLine();
        } else {
            d3.json("./data/clean/clean_rents.json", (error, dataMap) => {
                this.dataMap = dataMap;
                // create autofill box
                var zipcodes = Object.keys(this.dataMap).sort();
                autocomplete(document.getElementById("ZIPCodeInput"), zipcodes);
            });
            d3.json("./data/clean/clean_rents_20001.json", (error, dataInit) => {
                if (document.getElementById("ZIPCodeInput").value == "") {
                    document.getElementById("ZIPCodeInput").value = "20001"
                }

                document.getElementById("newZip").onclick = (e) => {this.newZIPCode()}

                var initialData = dataInit;
                this.initGraph(initialData)
                this.drawLegend();
                this.insertText();
                this.drawVerticalLine();
            });
        }
    }
}

// Graph.drawGraph()
// $(window).resize(function() {
//     Graph.drawGraph()
// })
new pym.Child();
pym.Child({ renderCallback: (containerWidth) =>{
    Graph.drawGraph(containerWidth)
}});

function autocomplete(inp, arr) {
  /*the autocomplete function takes two arguments,
  the text field element and an array of possible autocompleted values:*/
  var currentFocus;
  /*execute a function when someone writes in the text field:*/
  inp.addEventListener("input", function(e) {
      var a, b, i, val = this.value;
      /*close any already open lists of autocompleted values*/
      closeAllLists();
      if (!val) { return false;}
      currentFocus = -1;
      /*create a DIV element that will contain the items (values):*/
      a = document.createElement("DIV");
      a.setAttribute("id", this.id + "autocomplete-list");
      a.setAttribute("class", "autocomplete-items");
      /*append the DIV element as a child of the autocomplete container:*/
      this.parentNode.appendChild(a);
      /*for each item in the array...*/
      for (i = 0; i < arr.length; i++) {
        /*check if the item starts with the same letters as the text field value:*/
        if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
          /*create a DIV element for each matching element:*/
          b = document.createElement("DIV");
          /*make the matching letters bold:*/
          b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
          b.innerHTML += arr[i].substr(val.length);
          /*insert a input field that will hold the current array item's value:*/
          b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
          /*execute a function when someone clicks on the item value (DIV element):*/
              b.addEventListener("click", function(e) {
              /*insert the value for the autocomplete text field:*/
              inp.value = this.getElementsByTagName("input")[0].value;
              Graph.newZIPCode()
              /*close the list of autocompleted values,
              (or any other open lists of autocompleted values:*/
              closeAllLists();
          });
          a.appendChild(b);
        }
      }
  });
  /*execute a function presses a key on the keyboard:*/
  inp.addEventListener("keydown", function(e) {
      var x = document.getElementById(this.id + "autocomplete-list");
      if (x) x = x.getElementsByTagName("div");
      if (e.keyCode == 40) {
        /*If the arrow DOWN key is pressed,
        increase the currentFocus variable:*/
        currentFocus++;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 38) { //up
        /*If the arrow UP key is pressed,
        decrease the currentFocus variable:*/
        currentFocus--;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 13) {
        /*If the ENTER key is pressed, prevent the form from being submitted,*/
        e.preventDefault();
        if (currentFocus > -1) {
          /*and simulate a click on the "active" item:*/
          if (x) x[currentFocus].click();
        }
      }
  });
  function addActive(x) {
    /*a function to classify an item as "active":*/
    if (!x) return false;
    /*start by removing the "active" class on all items:*/
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    /*add class "autocomplete-active":*/
    x[currentFocus].classList.add("autocomplete-active");
  }
  function removeActive(x) {
    /*a function to remove the "active" class from all autocomplete items:*/
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }
  function closeAllLists(elmnt) {
    /*close all autocomplete lists in the document,
    except the one passed as an argument:*/
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
      x[i].parentNode.removeChild(x[i]);
    }
  }
}
/*execute a function when someone clicks in the document:*/
document.addEventListener("click", function (e) {
    closeAllLists(e.target);
});
}
