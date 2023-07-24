package com.shania.rsabuild1

import android.Manifest
import android.R.attr.bitmap
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import android.os.AsyncTask
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.net.Socket


const val STATUS_CODE = 0
var serverIp = "10.77.208.243"
var isPhotoTaken = false
private const val FILE_NAME = "rsa_photo.jpg"
private const val REQUEST_CODE = 42
private lateinit var photoFile: File

class MainActivity : AppCompatActivity() {
    private lateinit var etServerIp: EditText
    private lateinit var btnConnect: ImageButton
    private lateinit var imageView: ImageView
    private lateinit var btnTakePhoto: Button
    private lateinit var btnCallRobot: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        etServerIp = findViewById(R.id.etServerIp)
        btnConnect = findViewById((R.id.btnConnect))
        imageView = findViewById(R.id.imageView)
        btnTakePhoto = findViewById(R.id.btnTakePhoto)
        btnCallRobot = findViewById(R.id.btnCallRobot)

        btnConnect.setOnClickListener {
            Log.i("MainActivity", serverIp)
            if (etServerIp.text.isNotEmpty()) {
                serverIp = etServerIp.text.toString()
                Toast.makeText(this, "Connection established!", Toast.LENGTH_SHORT).show()
                Log.i("MainActivity", serverIp)
            }
            else {
                Toast.makeText(this, "Connection failed...", Toast.LENGTH_LONG).show()
            }
            // TODO ADD CONNECTION TEST AND TOAST ON BUTTON PRESS
        }

        btnTakePhoto.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            photoFile = getPhotoFile(FILE_NAME)

            val fileProvider = FileProvider.getUriForFile(this, "com.shania.rsabuild1.fileprovider", photoFile)
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            if (takePictureIntent.resolveActivity(this.packageManager) != null) {
                startActivityForResult(takePictureIntent, REQUEST_CODE)
            } else {
                Toast.makeText(this, "Unable to open Camera", Toast.LENGTH_SHORT).show()
            }
        }

        btnCallRobot.setOnClickListener {
            if (isPhotoTaken) {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED
                ) {
                    ActivityCompat.requestPermissions(
                        this,
                        arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                        STATUS_CODE
                    )
                } else {
                    // all permissions enabled
                    val sendcode = Send()
                    Log.i("MainActivity", "Socket initiated")
                    sendcode.execute()

                }
            }
            else {
                Toast.makeText(this, "Take a photo first!", Toast.LENGTH_SHORT).show()
                }
        }
    }

    private fun getPhotoFile(fileName: String): File {
        val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(fileName, ".jpg", storageDirectory)
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE && resultCode == RESULT_OK) {
            isPhotoTaken = true
            val takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)
            // Check Exif data to correct rotation
            val ei = ExifInterface(photoFile.absolutePath)
            val orientation: Int = ei.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            )

            //val rotatedBitmap = rotateBitmap(takenImage, 90f)
            var rotatedBitmap: Bitmap? = null
            rotatedBitmap = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(takenImage, 90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(takenImage, 180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(takenImage, 270f)
                ExifInterface.ORIENTATION_NORMAL -> takenImage
                else -> takenImage
            }

            imageView.setImageBitmap(rotatedBitmap)
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
    }
}

//Main Actions - Asynchronous
class Send : AsyncTask<Void?, Void?, Void?>() {
    companion object {
        private lateinit var s: Socket // Socket Variable
    }
    override fun doInBackground(vararg params: Void?): Void? {
        try {

            s = Socket(serverIp, 12345) // Connects to IP address - enter your IP here

            val input: InputStream = FileInputStream(photoFile.absolutePath)  // Gets the true path of image
            try {
                try {
                    // Reads bytes (all together)
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        s.getOutputStream().write(buffer, 0, bytesRead)  // Writes bytes to output stream
                    }
                } finally {
                    // Flushes and closes socket
                    s.getOutputStream().flush()
                    s.close()
                }
            } finally {
                input.close()
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return null
    }
}
