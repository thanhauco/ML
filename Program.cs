using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;

public class EmailData
{
    public string? Text { get; set; }
    public bool IsSpam { get; set; }
}

public class Program
{
    public static void Main(string[] args)
    {
        string dataPath = "emails.csv";

        try
        {
            // Manually parse and validate the CSV file
            var emailData = ParseAndValidateCsv(dataPath);

            var mlContext = new MLContext(seed: 0);

            // Convert our list to an IDataView
            var data = mlContext.Data.LoadFromEnumerable(emailData);

            // Split data into training and testing sets
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = trainTestData.TrainSet;
            var testingData = trainTestData.TestSet;

            Console.WriteLine($"Number of training samples: {trainingData.GetRowCount()}");
            Console.WriteLine($"Number of testing samples: {testingData.GetRowCount()}");

            // Define the training pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "IsSpam", featureColumnName: "Features"));

            // Train the model
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainingData);

            // Make predictions on the test set
            var predictions = model.Transform(testingData);

            // Evaluate the model
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "IsSpam");

            // Print the evaluation metrics
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:P2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:P2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:P2}");

            // Use the model to make predictions on new data
            var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, SpamPrediction>(model);

            // Example predictions
            var testEmails = new[]
            {
                new EmailData { Text = "Buy one, get one free!" },
                new EmailData { Text = "Meeting scheduled for tomorrow at 2 PM" }
            };

            foreach (var email in testEmails)
            {
                var prediction = predictionEngine.Predict(email);
                Console.WriteLine($"Email: {email.Text}");
                Console.WriteLine($"Predicted as {(prediction.IsSpam ? "spam" : "not spam")} with probability {prediction.Probability:P2}");
                Console.WriteLine();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    private static List<EmailData> ParseAndValidateCsv(string path)
    {
        var emailData = new List<EmailData>();
        var lines = File.ReadAllLines(path);

        var csvRegex = new Regex(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");

        for (int i = 1; i < lines.Length; i++) // Assuming the first line is a header
        {
            var line = lines[i];
            var parts = csvRegex.Split(line);

            if (parts.Length != 2)
            {
                throw new FormatException($"Invalid format in line {i + 1}: {line}");
            }

            var text = parts[0].Trim('"');
            bool isSpam;

            if (bool.TryParse(parts[1], out isSpam))
            {
                // The value is already a valid boolean
            }
            else if (int.TryParse(parts[1], out int intValue))
            {
                // The value is a number, treat 1 as true and 0 as false
                isSpam = intValue != 0;
            }
            else
            {
                throw new FormatException($"Invalid boolean value in line {i + 1}: {parts[1]}");
            }

            emailData.Add(new EmailData { Text = text, IsSpam = isSpam });
        }

        return emailData;
    }
}

public class SpamPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}