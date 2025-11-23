import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message_(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.real(np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector))
    decrypted_vector = decrypted_vector.round().astype(int)
    message = "".join([chr(num) for num in decrypted_vector])
    return message

message = input("Enter the message: ")
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted_message = encrypt_message(message, key_matrix)
decrypted_message = decrypt_message_(encrypted_message, key_matrix)
print("Encrypted message: ", encrypted_message)
print("Decrypted message: ", decrypted_message)