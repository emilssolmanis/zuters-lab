import gv

def __graphviz_tree(tree, names, graph, curr_node_handle):
    for node in tree.nodes:
        if node.attribute:
            label = names[node.attribute][0]
        else:
            label = names[0][node.node_class]
        edge_label = names[tree.attribute][node.value]
        gv_child = gv.node(graph, label + str(id(node)))
        gv.setv(gv_child, "label", label)
        gv_edge = gv.edge(curr_node_handle, gv_child)
        gv.setv(gv_edge, "label", edge_label)
        __graphviz_tree(node, names, graph, gv_child)


def render_graphviz(filename, tree, names):
    gv_graph = gv.digraph("ID3")
    gv_root = gv.node(gv_graph, names[tree.attribute][0])

    gv_tree = __graphviz_tree(tree, names, gv_graph, gv_root)

    gv.layout(gv_graph, "dot")
    return gv.render(gv_graph, "png", filename)


def test():
    import os
    import lab1

    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB1------------------------------------------"
    print "--------------------------------------------------------------------------------"

    lab1_root = os.path.dirname(__file__)
    samples = lab1.read_table(lab1_root + "/examples.txt")
    tree = lab1.id3(samples, {1, 2, 3, 4})
    for i, sample in enumerate(samples):
        assert sample[0] == lab1.classify(tree, sample)
    print "LAB1 tests: OK"

    names = []
    with open(lab1_root + "/attributevaluenames.txt") as f:
        for line in f:
            names.append(line.split())
    
    render_graphviz(lab1_root + "/tree.png", tree, names)
