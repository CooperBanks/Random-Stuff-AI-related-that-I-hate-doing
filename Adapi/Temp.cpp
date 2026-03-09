struct Task {
    std::string type;
    std::vector<float> features;
};

struct WorkerNode {
    NeuralNet net;
    std::string specialization;
    int currentLoad;

    void execute(Task t) {
        net.forward(t.features);
        adapt(t); // incrementally adjust
    }

    void adapt(Task t) {
        // increase neurons/layers if load > threshold
    }
};

struct ManagerNode {
    std::vector<WorkerNode*> workers;

    void assignTask(Task t) {
        WorkerNode* best = findWorker(t);
        if(best) best->execute(t);
        else spawnWorker(t);
    }

    WorkerNode* findWorker(Task t) {
        // compare similarity, return best
    }

    void spawnWorker(Task t) {
        WorkerNode* newWorker = new WorkerNode();
        newWorker->specialization = t.type;
        newWorker->net = NeuralNet(/*small initial network*/);
        workers.push_back(newWorker);
        newWorker->execute(t);
    }
};
