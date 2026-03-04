//THIS IS MADE AND DEVELOPED BY THE OWNERS OF THIS REPOSITORY https://github.com/CooperBanks/Random-Stuff-AI-related-that-I-hate-doing IF YOU FIND THIS ELSEWHERE IT IS FAKE

#include <crow.h>
#include "lstm.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

constexpr int vocab_size = 274;
constexpr int hidden_dim = 16;

// simple one-hot embedding <-- I Don't think this was ever implemented
Eigen::MatrixXd embeddings = Eigen::MatrixXd::Identity(vocab_size, vocab_size);

int main() {
    LSTM lstm(vocab_size, hidden_dim);

    crow::SimpleApp app;

    // Poorly designed web interface
    CROW_ROUTE(app, "/")([]() {
        return R"HTML(
        <!DOCTYPE html>
        <html>
        <head>
            <title>tiny_llm Neural Map</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.1/cytoscape.min.js"></script>
            <style>
                #cy { width: 100%; height: 600px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <h1>Neural Pathway Map</h1>
            <input id="char" type="text" maxlength="1" value="a"/>
            <button onclick="step()">Step</button>
            <div id="cy"></div>

            <script>
            let cy = cytoscape({ container: document.getElementById('cy'), elements: [], style: [
                { selector: 'node', style: { 'label': 'data(id)', 'background-color': '#0074D9', 'width': 30, 'height': 30 } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#aaa', 'target-arrow-color': '#aaa', 'target-arrow-shape': 'triangle' } }
            ], layout: { name: 'grid' }});

            async function step() {
                const c = document.getElementById('char').value;
                const res = await fetch('/step?c=' + encodeURIComponent(c));
                const data = await res.json();

                cy.elements().remove();

                // Add hidden neurons as nodes
                data.h.forEach((val, i) => {
                    cy.add({ group: 'nodes', data: { id: 'h'+i, value: val } });
                });

                // Add output neurons as nodes
                data.probs.forEach((val, i) => {
                    cy.add({ group: 'nodes', data: { id: 'v'+i, value: val } });
                });

                // Add edges: simple fully-connected hidden -> output for visualization
                data.h.forEach((_, i) => {
                    data.probs.forEach((_, j) => {
                        cy.add({ group: 'edges', data: { id: 'e'+i+'-'+j, source: 'h'+i, target: 'v'+j } });
                    });
                });

                cy.layout({ name: 'cose', animate: true }).run();
            }
            </script>
        </body>
        </html>
        )HTML";
    });

    // Step route
    CROW_ROUTE(app, "/step")([&lstm](const crow::request& req){
        crow::json::wvalue result;

        std::string c = req.url_params.get("c") ? req.url_params.get("c") : "a";
        int idx = static_cast<int>(c[0]) % vocab_size;
        Eigen::VectorXd x = embeddings.row(idx).transpose();

        Eigen::VectorXd probs = lstm.forward(x);

        std::vector<double> h_vals(lstm.hidden_dim);
        for(int i=0; i<lstm.hidden_dim; i++)
            h_vals[i] = lstm.h(i);

        std::vector<double> prob_vals(probs.size());
        for(int i=0; i<probs.size(); i++)
            prob_vals[i] = probs(i);

        result["h"] = h_vals;
        result["probs"] = prob_vals;

        // std::vector<double> weights; <-- Do something about this for V2

        return result;
    });

    std::cout << "Server running on http://0.0.0.0:8080\n"; // Should get a domain or something IDK
    app.port(8080).multithreaded().run();
}
