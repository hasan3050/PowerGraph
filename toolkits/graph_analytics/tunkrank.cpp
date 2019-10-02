/*
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */

// TunkRank algorithm based on http://thenoisychannel.com/2009/01/13/a-twitter-analog-to-pagerank/

#include <vector>
#include <string>
#include <fstream>

#include <graphlab.hpp>
// #include <graphlab/macros_def.hpp>

float RETWEET_PROB = 0.05;

float TOLERANCE = 1.0E-2;

size_t ITERATIONS = 0;
// The vertex data is just the tunkrank value (a float) which represents the vertex's influence
typedef float vertex_data_type;

// There is no edge data in the tunkrank application
typedef graphlab::empty edge_data_type;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

/*
 * A simple function used by graph.transform_vertices(init_vertex);
 * to initialize the vertes data.
 */
void init_vertex(graph_type::vertex_type& vertex) { vertex.data() = 1; }

class tunkrank :
  public graphlab::ivertex_program<graph_type, float>,
  public graphlab::IS_POD_TYPE {
  float last_change;
public:
  /* Gather the weighted influence of the vertex's followers   */
  float gather(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    return (1+RETWEET_PROB * edge.source().data()) / edge.source().num_out_edges();
  }

  /* Use the total influence of followers to update this vertex's influence */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& total) {
    last_change = std::fabs(total - vertex.data());
    vertex.data() = total;		//influence of this vertex
    if (ITERATIONS) context.signal(vertex);
  }

  /* The scatter edges depend on whether the tunkrank has converged */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // If an iteration counter is set then 
    if (ITERATIONS) return graphlab::NO_EDGES;
    if (last_change > TOLERANCE) return graphlab::OUT_EDGES;
    else return graphlab::NO_EDGES;
  }

  /* The scatter function just signal adjacent vertices */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    context.signal(edge.target());
  }

  void save(graphlab::oarchive& oarc) const {
    if (ITERATIONS == 0) oarc << last_change;
  }
  void load(graphlab::iarchive& iarc) {
    if (ITERATIONS == 0) iarc >> last_change;
  }

}; // end of factorized_tunkrank update functor


/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", tunkrank_writer()) to save the graph.
 */
struct tunkrank_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    std::stringstream strm;
    strm << v.id() << "\t" << v.data() << "\n";
    return strm.str();
  }
  std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of tunkrank writer



int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_INFO);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("TunkRank algorithm.");
  std::string graph_dir;
  std::string format = "adj";
  std::string exec_type = "synchronous";
  clopts.attach_option("graph", graph_dir,
                       "The graph file.  If none is provided "
                       "then a toy graph will be created");
  clopts.add_positional("graph");
  clopts.attach_option("engine", exec_type, 
                       "The engine type synchronous or asynchronous");
  clopts.attach_option("tol", TOLERANCE,
                       "The permissible change at convergence.");
  clopts.attach_option("format", format,
                       "The graph file format");
  size_t powerlaw = 0;
  clopts.attach_option("powerlaw", powerlaw,
                       "Generate a synthetic powerlaw out-degree graph. ");
  clopts.attach_option("iterations", ITERATIONS, 
                       "If set, will force the use of the synchronous engine"
                       "overriding any engine option set by the --engine parameter. "
                       "Runs complete (non-dynamic) PageRank for a fixed "
                       "number of iterations. Also overrides the iterations "
                       "option in the engine");
  std::string saveprefix;
  clopts.attach_option("saveprefix", saveprefix,
                       "If set, will save the resultant tunkrank to a "
                       "sequence of files with prefix saveprefix");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }


  if (ITERATIONS) {
    // make sure this is the synchronous engine
    dc.cout() << "--iterations set. Forcing Synchronous engine, and running "
              << "for " << ITERATIONS << " iterations." << std::endl;
    clopts.get_engine_args().set_option("type", "synchronous");
    clopts.get_engine_args().set_option("max_iterations", ITERATIONS);
    clopts.get_engine_args().set_option("sched_allv", true);
  }

  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  if(powerlaw > 0) { // make a synthetic graph
    dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
    graph.load_synthetic_powerlaw(powerlaw, false, 2, 100000000);
  }
  else if (graph_dir.length() > 0) { // Load the graph from a file
    dc.cout() << "Loading graph in format: "<< format << std::endl;
    graph.load_format(graph_dir, format);
  }
  else {
    dc.cout() << "graph or powerlaw option must be specified" << std::endl;
    clopts.print_description();
    return 0;
  }
  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << "#vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;

  // Initialize the vertex data
  graph.transform_vertices(init_vertex);

  // Running The Engine -------------------------------------------------------
  graphlab::omni_engine<tunkrank> engine(dc, graph, exec_type, clopts);
  engine.signal_all();
  engine.start();
  const float runtime = engine.elapsed_seconds();
  dc.cout() << "Finished Running engine in " << runtime
            << " seconds." << std::endl;

  // Save the final graph -----------------------------------------------------
  if (saveprefix != "") {
    graph.save(saveprefix, tunkrank_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
  }

  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main



