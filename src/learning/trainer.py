'''
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_gnn(network, signals, epochs=100):
    logger.info("Preparing data for GNN training...")
    # Prepare the data
    node_features = torch.tensor([signals[i] for i in range(len(network.get_nodes()))], dtype=torch.float)
    edge_index = torch.tensor(list(network.graph.edges()), dtype=torch.long).t().contiguous()
    data = Data(x=node_features, edge_index=edge_index)
    logger.info(f"Data prepared: {data}")

    # Initialize the model
    logger.info("Initializing GNN model...")
    model = GNN(in_channels=data.num_node_features, hidden_channels=16, out_channels=data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    logger.info(f"Model initialized: {model}")

    # Training loop
    logger.info("Starting GNN training...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)  # Simple autoencoder-like loss for now
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    logger.info("GNN training finished.")
    return model
'''