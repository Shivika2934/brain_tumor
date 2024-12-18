$(document).ready(function () {
    // Function to handle file input change
    $("#imageUpload").change(function () {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            $('.image-section').show();  // Show image section
        };

        if (this.files && this.files[0]) {
            reader.readAsDataURL(this.files[0]);

            // Show Predict button after file is chosen
            $("#btn-predict").show();
        } else {
            alert("Please select a file first!");
        }
    });

    // Handle predict button click
    $("#btn-predict").click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();  // Hide Predict button
        $('.loader').show();  // Show loader

        // Make prediction by calling the Flask backend
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (result) {
                $('.loader').hide();  // Hide loader
                $('#btn-predict').show();  // Show Predict button again
                $('#result').text(result);  // Display result
            },
            error: function (error) {
                console.log(error);
                $('.loader').hide();
                $('#btn-predict').show();
                alert('Error in prediction. Please try again.');
            }
        });
    });
});
