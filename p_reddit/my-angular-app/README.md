# My Angular App

This project is an Angular application that integrates with a Django backend API defined in `views.py`. The application provides various functionalities such as word counting, text evaluation, and frequency analysis.

## Project Structure

```
my-angular-app
├── src
│   ├── app
│   │   ├── services
│   │   │   └── api.service.ts
│   │   ├── app.module.ts
│   │   └── app.component.ts
├── angular.json
├── package.json
└── README.md
```

## Features

- **Word Count**: Calls the API to count words in a given text.
- **Text Evaluation**: Evaluates the provided text and returns the result.
- **Depressed Words**: Retrieves a list of words associated with depression.
- **Post Frequency**: Analyzes and returns the frequency of posts.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd my-angular-app
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Running the Application

To start the application, run the following command:
```
ng serve
```
Then, open your browser and navigate to `http://localhost:4200`.

## API Integration

The application integrates with the following API endpoints defined in the Django backend:

- `GET /word_count`: Returns the word count.
- `GET /evaluate_text?text=<text>`: Evaluates the provided text.
- `GET /depressed_words`: Returns a list of depressed words.
- `GET /post_frequency`: Returns the frequency of posts.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.