package simulator;

import javax.swing.SwingUtilities;

public class Main {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            Simulator simulator = Simulator.getInstance();
            simulator.runSimulation();
        });
    }
}