import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;

public class VQA_UI {
    private JFrame frame;
    private JLabel imageLabel;
    private JTextField questionField;
    private JButton submitButton;
    private JLabel answerLabel;
    private File selectedFile;

    public VQA_UI() {
        frame = new JFrame("VQA Model UI");
        frame.setSize(500, 400);
        frame.setLayout(new BorderLayout());
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel topPanel = new JPanel();
        JButton uploadButton = new JButton("Select Image");
        uploadButton.addActionListener(e -> selectImage());
        topPanel.add(uploadButton);
        
        imageLabel = new JLabel("No Image Selected", SwingConstants.CENTER);

        JPanel middlePanel = new JPanel();
        questionField = new JTextField(30);
        middlePanel.add(new JLabel("Enter Question: "));
        middlePanel.add(questionField);
        
        submitButton = new JButton("Ask VQA Model");
        submitButton.addActionListener(e -> sendDataToModel());
        middlePanel.add(submitButton);
        
        answerLabel = new JLabel("Answer: ", SwingConstants.CENTER);

        frame.add(topPanel, BorderLayout.NORTH);
        frame.add(imageLabel, BorderLayout.CENTER);
        frame.add(middlePanel, BorderLayout.SOUTH);
        frame.add(answerLabel, BorderLayout.PAGE_END);
        
        frame.setVisible(true);
    }

    private void selectImage() {
        JFileChooser fileChooser = new JFileChooser();
        int returnValue = fileChooser.showOpenDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
            imageLabel.setText("Selected: " + selectedFile.getName());
        }
    }

    private void sendDataToModel() {
        if (selectedFile == null || questionField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please select an image and enter a question.", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        String question = questionField.getText();
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get(selectedFile.getAbsolutePath()));
            String base64Image = Base64.getEncoder().encodeToString(imageBytes);
            
            String response = callVQAModel(base64Image, question);
            answerLabel.setText("Answer: " + response);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String callVQAModel(String base64Image, String question) {
        return "This is a sample response.";
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(VQA_UI::new);
    }
}